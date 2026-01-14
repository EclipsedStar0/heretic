# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>
# THIS IS src/heretic/main.py

import os

# Disable torch.compile on Windows to avoid Unicode and compiler issues
if os.name == 'nt':  # Windows
    os.environ['TORCHDYNAMO_DISABLE'] = '1'



from datetime import datetime
import pickle
import json
import pandas as pd

import math
import os
import sys
import time
import warnings
from importlib.metadata import version
from os.path import commonprefix
from pathlib import Path

import huggingface_hub
import optuna
import torch
import torch.nn.functional as F
import transformers
from accelerate.utils import (
    is_mlu_available,
    is_musa_available,
    is_npu_available,
    is_sdaa_available,
    is_xpu_available,
)
from huggingface_hub import ModelCard, ModelCardData
from optuna import Trial, TrialPruned
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import TPESampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
from pydantic import ValidationError
from questionary import Choice
from rich.traceback import install
from transformers import AutoModelForCausalLM

from .analyzer import Analyzer
from .config import QuantizationMethod, Settings
from .evaluator import Evaluator
from .model import AbliterationParameters, Model
from .utils import (
    empty_cache,
    format_duration,
    get_readme_intro,
    get_trial_parameters,
    load_prompts,
    print,
    prompt_password,
    prompt_path,
    prompt_select,
    prompt_text,
)

def get_trial_parameters_objects(trial: Trial) -> dict[str, AbliterationParameters]:
    """Get AbliterationParameters objects from a trial."""
    if "parameters" not in trial.user_attrs:
        return {}
    
    parameters_dict = trial.user_attrs["parameters"]
    return {
        component: AbliterationParameters(**params_dict)
        for component, params_dict in parameters_dict.items()
    }


def obtain_merge_strategy(settings: Settings) -> str | None:
    """
    Prompts the user for how to proceed with quantized models.
    Returns "merge", "adapter", or None (if cancelled/invalid).
    """
    # Prompt for all PEFT models to ensure user is aware of merge implications
    if settings.quantization in [QuantizationMethod.BNB_4BIT, QuantizationMethod.GPTQ, QuantizationMethod.AGPTQ]:
        # Quantized models need special handling - we must reload the base model
        # in full precision to merge the LoRA adapters
        print()
        print(
            "[yellow]Model was loaded with quantization. Merging requires reloading the base model.[/]"
        )
        if settings.quantization in [QuantizationMethod.GPTQ, QuantizationMethod.AGPTQ]:
            print(
                "[yellow]Note: GPTQ models can stay quantized during merge, reducing memory requirements.[/]"
            )
        else:
            print(
                "[red](!) WARNING: CPU Merging requires dequantizing the entire model to System RAM.[/]"
            )
            print("[red]    This can lead to SYSTEM FREEZES if you run out of memory.[/]")
        print(
            "[yellow]    Rule of thumb: You need approx. 3x the parameter count in GB.[/]"
        )

        try:
            # Estimate memory requirements by loading the model structure on the "meta" device.
            # This doesn't consume actual RAM but allows us to inspect the parameter count/dtype.
            #
            # Suppress warnings during meta device loading (e.g., "Some weights were not initialized").
            # These are expected and harmless since we're only inspecting model structure, not running inference.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                meta_model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    device_map="meta",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
                footprint_bytes = meta_model.get_memory_footprint()
                footprint_gb = footprint_bytes / (1024**3)
                print(
                    f"[yellow]    Estimated net RAM required for model weights (excluding overhead): [bold]~{footprint_gb:.1f} GB[/][/]"
                )
        except Exception:
            # Fallback if meta loading fails (e.g. owing to custom model code
            # or `bitsandbytes` quantization config issues on the meta device)
            print(
                "[yellow]    Example: A 27B model requires ~80GB RAM. A 70B model requires ~200GB RAM.[/]"
            )
        print()

        merge_choice = prompt_select(
            "How do you want to proceed?",
            choices=[
                Choice(
                    title="Merge full model (reload base model on CPU - requires high RAM)",
                    value="merge",
                ),
                Choice(
                    title="Save LoRA adapter only (can be merged later with llama.cpp or more RAM)",
                    value="adapter",
                ),
            ],
        )
        return merge_choice

    # Default for non-quantized models
    return "merge"


def run():
    # Enable expandable segments to reduce memory fragmentation on multi-GPU setups.
    if (
        "PYTORCH_ALLOC_CONF" not in os.environ
        and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    ):
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Modified "Pagga" font from https://budavariam.github.io/asciiart-text/
    print("Start Time: ", datetime.now())
    print(f"[cyan]█░█░█▀▀░█▀▄░█▀▀░▀█▀░█░█▀▀[/]  v{version('heretic-llm')}")
    print("[cyan]█▀█░█▀▀░█▀▄░█▀▀░░█░░█░█░░[/]")
    print(
        "[cyan]▀░▀░▀▀▀░▀░▀░▀▀▀░░▀░░▀░▀▀▀[/]  [blue underline]https://github.com/p-e-w/heretic[/]"
    )
    print()

    if (
        # There is at least one argument (argv[0] is the program name).
        len(sys.argv) > 1
        # No model has been explicitly provided.
        and "--model" not in sys.argv
        # The last argument is a parameter value rather than a flag (such as "--help").
        and not sys.argv[-1].startswith("-")
    ):
        # Assume the last argument is the model.
        sys.argv.insert(-1, "--model")

    try:
        # The required argument "model" must be provided by the user,
        # either on the command line or in the configuration file.
        settings = Settings()  # ty:ignore[missing-argument]
    except ValidationError as error:
        print(f"[red]Configuration contains [bold]{error.error_count()}[/] errors:[/]")

        for error in error.errors():
            print(f"[bold]{error['loc'][0]}[/]: [yellow]{error['msg']}[/]")

        print()
        print(
            "Run [bold]heretic --help[/] or see [bold]config.default.toml[/] for details about configuration parameters."
        )
        return

    # Adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/env.py
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        print(f"Detected [bold]{count}[/] CUDA device(s):")
        for i in range(count):
            print(f"* GPU {i}: [bold]{torch.cuda.get_device_name(i)}[/]")
    elif is_xpu_available():
        count = torch.xpu.device_count()
        print(f"Detected [bold]{count}[/] XPU device(s):")
        for i in range(count):
            print(f"* XPU {i}: [bold]{torch.xpu.get_device_name(i)}[/]")
    elif is_mlu_available():
        count = torch.mlu.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MLU device(s):")
        for i in range(count):
            print(f"* MLU {i}: [bold]{torch.mlu.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_sdaa_available():
        count = torch.sdaa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] SDAA device(s):")
        for i in range(count):
            print(f"* SDAA {i}: [bold]{torch.sdaa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_musa_available():
        count = torch.musa.device_count()  # ty:ignore[unresolved-attribute]
        print(f"Detected [bold]{count}[/] MUSA device(s):")
        for i in range(count):
            print(f"* MUSA {i}: [bold]{torch.musa.get_device_name(i)}[/]")  # ty:ignore[unresolved-attribute]
    elif is_npu_available():
        print(f"NPU detected (CANN version: [bold]{torch.version.cann}[/])")  # ty:ignore[unresolved-attribute]
    elif torch.backends.mps.is_available():
        print("Detected [bold]1[/] MPS device (Apple Metal)")
    else:
        print(
            "[bold yellow]No GPU or other accelerator detected. Operations will be slow.[/]"
        )

    # We don't need gradients as we only do inference.
    torch.set_grad_enabled(False)

    # While determining the optimal batch size, we will try many different batch sizes,
    # resulting in many computation graphs being compiled. Raising the limit (default = 8)
    # avoids errors from TorchDynamo assuming that something is wrong because we
    # recompile too often.
    torch._dynamo.config.cache_size_limit = 64

    # Silence warning spam from Transformers.
    # In my entire career I've never seen a useful warning from that library.
    transformers.logging.set_verbosity_error()

    # We do our own trial logging, so we don't need the INFO messages
    # about parameters and results.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Silence the warning about multivariate TPE being experimental.
    warnings.filterwarnings("ignore", category=ExperimentalWarning)

    model = Model(settings)

    print()
    print(f"Loading good prompts from [bold]{settings.good_prompts.dataset}[/] at {datetime.now()}...")
    good_prompts = load_prompts(settings, settings.good_prompts)
    print(f"* [bold]{len(good_prompts)}[/] prompts loaded at {datetime.now()}")

    print()
    print(f"Loading bad prompts from [bold]{settings.bad_prompts.dataset}[/] at {datetime.now()}...")
    bad_prompts = load_prompts(settings, settings.bad_prompts)
    print(f"* [bold]{len(bad_prompts)}[/] prompts loaded at {datetime.now()}")

    if settings.batch_size == 0:
        print()
        print("Determining optimal batch size...")

        batch_size = 1
        best_batch_size = -1
        best_performance = -1

        while batch_size <= settings.max_batch_size:
            print(f"* Trying batch size [bold]{batch_size}[/] at {datetime.now()}... ", end="")

            prompts = good_prompts * math.ceil((batch_size / len(good_prompts)) / 10)
            prompts = prompts[:batch_size]

            try:
                # Warmup run to build the computation graph so that part isn't benchmarked.
                model.get_responses(prompts)

                start_time = time.perf_counter()
                responses = model.get_responses(prompts)
                end_time = time.perf_counter()
            except Exception as error:
                if batch_size == 1:
                    # Even a batch size of 1 already fails.
                    # We cannot recover from this.
                    raise

                print(f"[red]Failed[/] ({error})")
                break

            response_lengths = [
                len(model.tokenizer.encode(response)) for response in responses
            ]
            performance = sum(response_lengths) / (end_time - start_time)

            print(f"[green]Ok[/] ([bold]{performance:.0f}[/] tokens/s)")

            if performance > best_performance:
                best_batch_size = batch_size
                best_performance = performance

            batch_size *= 2

        settings.batch_size = best_batch_size
        print(f"* Chosen batch size: [bold]{settings.batch_size}[/]")

    print()
    print(f"Checking for common response prefix at {datetime.now()}...")
    responses = model.get_responses_batched(good_prompts[:100] + bad_prompts[:100])
    with open('dumps/prefix_responses.pkl', 'wb') as file:
        pickle.dump(responses, file)
    #print("Loading from file...")
    #with open("dumps/prefix_responses.pkl", 'rb') as file:
    #    responses = pickle.load(file)
    
    #print(f"[INFO]: Manually skipping commonprefix check at {datetime.now()}")
    #responses = []

    # Despite being located in os.path, commonprefix actually performs
    # a naive string operation without any path-specific logic,
    # which is exactly what we need here. Trailing spaces are removed
    # to avoid issues where multiple different tokens that all start
    # with a space character lead to the common prefix ending with
    # a space, which would result in an uncommon tokenization.
    
    model.response_prefix = commonprefix(responses).rstrip(" ")
    #model.response_prefix = ""

    # Suppress CoT output.
    if model.response_prefix.startswith("<think>"):
        # Most thinking models.
        model.response_prefix = "<think></think>"
    elif model.response_prefix.startswith("<|channel|>analysis<|message|>"):
        # gpt-oss.
        model.response_prefix = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
    elif model.response_prefix.startswith("<thought>"):
        # Unknown, suggested by user.
        model.response_prefix = "<thought></thought>"
    elif model.response_prefix.startswith("[THINK]"):
        # Unknown, suggested by user.
        model.response_prefix = "[THINK][/THINK]"

    if model.response_prefix:
        print(f"* Prefix found: [bold]{model.response_prefix!r}[/]")
    else:
        print("* None found")

    evaluator = Evaluator(settings, model)

    if settings.evaluate_model is not None:
        print()
        print(f"Loading model [bold]{settings.evaluate_model}[/] at {datetime.now()}...")
        settings.model = settings.evaluate_model
        model.reset_model()
        print("* Evaluating...")
        evaluator.get_score()
        return

    print()
    print("Calculating per-layer refusal directions...")
    print(f"* Obtaining residuals for good prompts at {datetime.now()}...")
    good_residuals = model.get_residuals_batched(good_prompts)
    torch.save(good_residuals, 'dumps/good_residuals.pt')
    #good_residuals = torch.load('dumps/good_residuals.pt')
    
    print(f"* Obtaining residuals for bad prompts at {datetime.now()}...")
    bad_residuals = model.get_residuals_batched(bad_prompts)
    torch.save(bad_residuals, 'dumps/bad_residuals.pt')
    #bad_residuals = torch.load('dumps/bad_residuals.pt')
        
    refusal_directions = F.normalize(
        bad_residuals.mean(dim=0) - good_residuals.mean(dim=0),
        p=2,
        dim=1,
    )

    analyzer = Analyzer(settings, model, good_residuals, bad_residuals)

    if settings.print_residual_geometry:
        analyzer.print_residual_geometry()

    if settings.plot_residuals:
        analyzer.plot_residuals()

    # We don't need the residuals after computing refusal directions.
    del good_residuals, bad_residuals, analyzer
    empty_cache()

    trial_index = 0
    start_time = time.perf_counter()

    def objective(trial: Trial) -> tuple[float, float]:
        nonlocal trial_index
        trial_index += 1
        trial.set_user_attr("index", trial_index)

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(model.get_layers()) - 1

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        parameters = {}

        for component in model.get_abliterable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            max_weight = trial.suggest_float(
                f"{component}.max_weight",
                0.8,
                1.5,
            )
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            # For sampling purposes, min_weight is expressed as a fraction of max_weight,
            # again because multivariate TPE doesn't support variable-range parameters.
            # The value is transformed into the actual min_weight value below.
            min_weight = trial.suggest_float(
                f"{component}.min_weight",
                0.0,
                1.0,
            )
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                0.6 * last_layer_index,
            )

            parameters[component] = AbliterationParameters(
                max_weight=max_weight,
                max_weight_position=max_weight_position,
                min_weight=(min_weight * max_weight),
                min_weight_distance=min_weight_distance,
            )

        trial.set_user_attr("direction_index", direction_index)
        
        parameters_dict = {
            component: params.to_dict()
            for component, params in parameters.items()
        }
        trial.set_user_attr("parameters", parameters_dict)

        print()
        print(
            f"Running trial [bold]{trial_index}[/] of [bold]{settings.n_trials}[/] at {datetime.now()}..."
        )
        print("* Parameters:")
        for name, value in get_trial_parameters(trial).items():
            print(f"  * {name} = [bold]{value}[/]")
        print(f"* Resetting model at {datetime.now()}...")
        model.reset_model()
        print(f"* Abliterating at {datetime.now()}...")
        parameters_objects = get_trial_parameters_objects(trial)
        model.abliterate(refusal_directions, direction_index, parameters_objects)
        print(f"* Evaluating at {datetime.now()}...")
        score, kl_divergence, refusals = evaluator.get_score()

        elapsed_time = time.perf_counter() - start_time
        remaining_time = (elapsed_time / trial_index) * (
            settings.n_trials - trial_index
        )
        print()
        print(f"[grey50]Elapsed time: [bold]{format_duration(elapsed_time)}[/][/]")
        if trial_index < settings.n_trials:
            print(
                f"[grey50]Estimated remaining time: [bold]{format_duration(remaining_time)}[/][/]"
            )

        trial.set_user_attr("kl_divergence", kl_divergence)
        trial.set_user_attr("refusals", refusals)

        return score

    def objective_wrapper(trial: Trial) -> tuple[float, float]:
        try:
            result = objective(trial)
            
            # Save trial results to disk
            save_trial_results(trial, settings)
            
            return result
        except KeyboardInterrupt:
            # Save before pruning
            try:
                if trial.state != TrialState.COMPLETE:
                    save_trial_results(trial, settings)
            except:
                pass
            trial.study.stop()
            raise TrialPruned()

    # Remove the conflicting backup loading code and just use the study loading from storage
    if settings.storage:
        try:
            study = optuna.load_study(
                study_name=settings.study_name or "heretic_study",
                storage=settings.storage,
                sampler=TPESampler(
                    n_startup_trials=settings.n_startup_trials,
                    n_ei_candidates=128,
                    multivariate=True,
                ),
            )
            print(f"[green]✓ Loaded existing study with {len(study.trials)} trials[/]")
            
            # Calculate remaining trials
            completed_trials = len([t for t in study.trials if t.state == TrialState.COMPLETE])
            remaining_trials = max(settings.n_trials - completed_trials, 0)
            trial_index = completed_trials
            
            if remaining_trials > 0 and hasattr(settings, 'resume_trials') and settings.resume_trials:
                print(f"[yellow]Resuming with {remaining_trials} additional trials to go[/]")
                settings.n_trials = completed_trials + remaining_trials
            else:
                print(f"[yellow]Using existing {completed_trials} completed trials[/]")
                
        except KeyError:
            # Study doesn't exist, create new
            study = optuna.create_study(
                study_name=settings.study_name or "heretic_study",
                storage=settings.storage,
                sampler=TPESampler(
                    n_startup_trials=settings.n_startup_trials,
                    n_ei_candidates=128,
                    multivariate=True,
                ),
                directions=[StudyDirection.MINIMIZE, StudyDirection.MINIMIZE],
            )
            print("[green]✓ Created new study[/]")
    else:
        # For in-memory studies, keep simple backup logic
        if os.path.exists("trials_backup.pkl"):
            try:
                with open("trials_backup.pkl", "rb") as f:
                    existing_trials = pickle.load(f)
                
                # Simple resume for in-memory studies
                for trial_data in existing_trials:
                    # Get the value from whichever field it's stored in
                    value = trial_data.get('value') or trial_data.get('score')
                    if value is None:
                        continue
                        
                    trial = optuna.trial.create_trial(
                        params=trial_data.get('params', {}),
                        distributions={k: optuna.distributions.FloatDistribution(0, 1) for k in trial_data.get('params', {})},
                        value=value,
                        user_attrs=trial_data.get('user_attrs', {}),
                    )
                    study.add_trial(trial)
                
                print(f"[green]Resumed {len(existing_trials)} trials from backup[/]")
            except Exception as e:
                print(f"[yellow]Could not load backup: {e}[/]")

    # And add this to periodically backup during optimization
    if trial_index % 5 == 0:  # Backup every 5 trials
        backup_trials = []
        for t in study.trials:
            if t.state == TrialState.COMPLETE:
                backup_trials.append({
                    'trial_id': t.number,
                    'user_attrs': dict(t.user_attrs),
                    'params': t.params,
                    'distributions': dict(t.distributions),
                    'datetime': datetime.now().isoformat(),
                })
        
        with open("trials_backup.pkl", "wb") as f:
            pickle.dump(backup_trials, f)

    def load_trials_from_files(study: optuna.study.Study, settings: Settings, trials_dir: Path = Path("trial_results")):
        """Manually add trials from saved files to a study."""
        if not trials_dir.exists():
            return
        
        summary_file = trials_dir / "trials_summary.csv"
        if not summary_file.exists():
            return
        
        df = pd.read_csv(summary_file)
        
        for _, row in df.iterrows():
            trial_file = trials_dir / f"trial_{int(row['trial_id']):04d}.json"
            if trial_file.exists():
                with open(trial_file, 'r') as f:
                    trial_data = json.load(f)
                
                # Check if trial already exists in study
                existing_trial = next((t for t in study.trials if t.number == trial_data['trial_id']), None)
                if existing_trial is None:
                    # Create a frozen trial and add it
                    frozen_trial = FrozenTrial(
                        number=trial_data['trial_id'],
                        state=TrialState[trial_data['state']],
                        params=trial_data['params'],
                        user_attrs=trial_data['user_attrs'],
                        system_attrs=trial_data['system_attrs'],
                        value=trial_data['value'],
                        datetime_start=None,
                        datetime_complete=None,
                        intermediate_values={},
                        distributions={},
                        trial_id=trial_data['trial_id'],
                    )
                    study.add_trial(frozen_trial)
        
        print(f"[green]✓ Loaded {len(study.trials)} trials from files[/]")

    def save_trial_results(trial: Trial, settings: Settings, output_dir: Path = Path("trial_results")):
        """Save trial results to disk for manual inspection/resume."""
        output_dir.mkdir(exist_ok=True)
        
        trial_data = {
            'trial_id': trial.number,
            'user_attrs': dict(trial.user_attrs),
            'params': trial.params,
            'distributions': dict(trial.distributions),
            'datetime': datetime.now().isoformat(),
        }
        
        # Save individual trial
        trial_file = output_dir / f"trial_{trial.number:04d}.json"
        with open(trial_file, 'w') as f:
            json.dump(trial_data, f, indent=2, default=str)
        
        # Update master summary
        summary_file = output_dir / "trials_summary.csv"
        
        if summary_file.exists():
            df = pd.read_csv(summary_file)
        else:
            df = pd.DataFrame()
        
        new_row = {
            'trial_id': trial.number,
            'index': trial.user_attrs.get('index', trial.number),
            'kl_divergence': trial.user_attrs.get('kl_divergence', None),
            'refusals': trial.user_attrs.get('refusals', None),
            'direction_index': trial.user_attrs.get('direction_index', None),
            'datetime': datetime.now().isoformat(),
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(summary_file, index=False)
        
        print(f"[grey50]Saved trial {trial.number} results[/]")


    BACKUP_FILE = "trials_backup.pkl"

    if os.path.exists(BACKUP_FILE):
        try:
            with open(BACKUP_FILE, "rb") as f:
                existing_trials = pickle.load(f)
            
            # Convert to FrozenTrials and add to study
            for i, trial_data in enumerate(existing_trials):
                frozen_trial = optuna.trial.FrozenTrial(
                    number=i,
                    state=optuna.trial.TrialState.COMPLETE,
                    value=trial_data.get('user_attrs', None),
                    values=(trial_data['score'], trial_data.get('kl_divergence', 0)),
                    params=trial_data['params'],
                    distributions={},
                    user_attrs=trial_data.get('user_attrs', {}),
                    system_attrs=trial_data.get('system_attrs', {}),
                    datetime_start=None,
                    datetime_complete=None,
                    intermediate_values={},
                    trial_id=i+1,
                )
                study.add_trial(frozen_trial)
            
            print(f"[green]✓ Resumed {len(existing_trials)} trials[/]")
        except Exception as e:
            print(f"[yellow]Could not load backup: {e}[/]")

    # Run optimization
    try:
        study.optimize(objective_wrapper, n_trials=settings.n_trials)
        
        # Backup after completion
        completed_trials = []
        for t in study.trials:
            if t.state == TrialState.COMPLETE:
                completed_trials.append({
                    'params': t.params,
                    'score': t.values[0] if t.values else None,
                    'user_attrs': dict(t.user_attrs),
                })
        
        with open(BACKUP_FILE, "wb") as f:
            pickle.dump(completed_trials, f)
            
    except KeyboardInterrupt:
        # Backup on interrupt
        print("\n[green]Backing up trials before exit...[/]")
        completed_trials = []
        for t in study.trials:
            if t.state == TrialState.COMPLETE:
                completed_trials.append({
                    'params': t.params,
                    'score': t.values[0] if t.values else None,
                    'user_attrs': dict(t.user_attrs),
                })
        
        with open(BACKUP_FILE, "wb") as f:
            pickle.dump(completed_trials, f)
        
        raise Exception

    while True:
        # If no trials at all have been evaluated, the study must have been stopped
        # by pressing Ctrl+C while the first trial was running. In this case, we just
        # re-raise the interrupt to invoke the standard handler defined below.
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if not completed_trials:
            raise KeyboardInterrupt

        # Get the Pareto front of trials. We can't use study.best_trials directly
        # as get_score() doesn't return the pure KL divergence and refusal count.
        # Note: Unlike study.best_trials, this does not handle objective constraints.
        sorted_trials = sorted(
            completed_trials,
            key=lambda trial: (
                trial.user_attrs["refusals"],
                trial.user_attrs["kl_divergence"],
            ),
        )
        min_divergence = math.inf
        best_trials = []
        for trial in sorted_trials:
            kl_divergence = trial.user_attrs["kl_divergence"]
            if kl_divergence < min_divergence:
                min_divergence = kl_divergence
                best_trials.append(trial)

        choices = [
            Choice(
                title=(
                    f"[Trial {trial.user_attrs['index']:>3}] "
                    f"Refusals: {trial.user_attrs['refusals']:>2}/{len(evaluator.bad_prompts)}, "
                    f"KL divergence: {trial.user_attrs['kl_divergence']:.4f}"
                ),
                value=trial,
            )
            for trial in best_trials
        ]

        choices.append(
            Choice(
                title="Continue optimization (run more trials)",
                value="continue",
            )
        )

        choices.append(
            Choice(
                title="None (exit program)",
                value="",
            )
        )

        print()
        print(f"[bold green]Optimization finished![/] at {datetime.now()}")
        print()
        print(
            (
                "The following trials resulted in Pareto optimal combinations of refusals and KL divergence. "
                "After selecting a trial, you will be able to save the model, upload it to Hugging Face, "
                "or chat with it to test how well it works. You can return to this menu later to select a different trial. "
                "[yellow]Note that KL divergence values above 1 usually indicate significant damage to the original model's capabilities.[/]"
            )
        )

        while True:
            print()
            trial = prompt_select("Which trial do you want to use?", choices)

            if trial == "continue":
                while True:
                    try:
                        n_more_trials = int(
                            prompt_text("How many more trials do you want to run?")
                        )
                        if n_more_trials > 0:
                            break
                        print("[red]Please enter a number greater than 0.[/]")
                    except ValueError:
                        print("[red]Invalid input. Please enter a number.[/]")

                settings.n_trials += n_more_trials
                try:
                    study.optimize(objective_wrapper, n_trials=n_more_trials)
                except KeyboardInterrupt:
                    pass
                break

            elif trial is None or trial == "":
                return

            print()
            print(f"Restoring model from trial [bold]{trial.user_attrs['index']}[/] at {datetime.now()}...")
            print("* Parameters:")
            for name, value in get_trial_parameters(trial).items():
                print(f"  * {name} = [bold]{value}[/]")
            print(f"* Resetting model at {datetime.now()}...")
            model.reset_model()
            print(f"* Abliterating at {datetime.now()}...")
            
            parameters_dict = trial.user_attrs["parameters"]
            parameters_objects = {
                component: AbliterationParameters(**params_dict)
                for component, params_dict in parameters_dict.items()
            }

            model.abliterate(
                refusal_directions,
                trial.user_attrs["direction_index"],
                parameters_objects,
            )

            while True:
                print()
                action = prompt_select(
                    "What do you want to do with the decensored model?",
                    [
                        "Save the model to a local folder",
                        "Upload the model to Hugging Face",
                        "Chat with the model",
                        "Nothing (return to trial selection menu)",
                    ],
                )

                if (
                    action is None
                    or action == "Nothing (return to trial selection menu)"
                ):
                    break

                # All actions are wrapped in a try/except block so that if an error occurs,
                # another action can be tried, instead of the program crashing and losing
                # the optimized model.
                try:
                    match action:
                        case "Save the model to a local folder":
                            save_directory = prompt_path("Path to the folder:")
                            if not save_directory:
                                continue

                            print("Saving model...")
                            strategy = obtain_merge_strategy(settings)
                            if strategy is None:
                                print("[yellow]Action cancelled.[/]")
                                continue

                            if strategy == "adapter":
                                model.model.save_pretrained(save_directory)
                            else:
                                merged_model = model.get_merged_model()
                                merged_model.save_pretrained(save_directory)
                                del merged_model
                                empty_cache()

                            model.tokenizer.save_pretrained(save_directory)
                            print(f"Model saved to [bold]{save_directory}[/].")

                        case "Upload the model to Hugging Face":
                            # We don't use huggingface_hub.login() because that stores the token on disk,
                            # and since this program will often be run on rented or shared GPU servers,
                            # it's better to not persist credentials.
                            token = huggingface_hub.get_token()
                            if not token:
                                token = prompt_password("Hugging Face access token:")
                            if not token:
                                continue

                            user = huggingface_hub.whoami(token)
                            fullname = user.get(
                                "fullname",
                                user.get("name", "unknown user"),
                            )
                            email = user.get("email", "no email found")
                            print(f"Logged in as [bold]{fullname} ({email})[/]")

                            repo_id = prompt_text(
                                "Name of repository:",
                                default=f"{user['name']}/{Path(settings.model).name}-heretic",
                            )

                            visibility = prompt_select(
                                "Should the repository be public or private?",
                                [
                                    "Public",
                                    "Private",
                                ],
                            )
                            private = visibility == "Private"

                            strategy = obtain_merge_strategy(settings)
                            if strategy is None:
                                print("[yellow]Action cancelled.[/]")
                                continue

                            if strategy == "adapter":
                                print("Uploading LoRA adapter...")
                                model.model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                            else:
                                print("Uploading merged model...")
                                merged_model = model.get_merged_model()
                                merged_model.push_to_hub(
                                    repo_id,
                                    private=private,
                                    token=token,
                                )
                                del merged_model
                                empty_cache()

                            model.tokenizer.push_to_hub(
                                repo_id,
                                private=private,
                                token=token,
                            )

                            # If the model path doesn't exist locally, it can be assumed
                            # to be a model hosted on the Hugging Face Hub, in which case
                            # we can retrieve the model card.
                            if not Path(settings.model).exists():
                                card = ModelCard.load(settings.model)
                                if card.data is None:
                                    card.data = ModelCardData()
                                if card.data.tags is None:
                                    card.data.tags = []
                                card.data.tags.append("heretic")
                                card.data.tags.append("uncensored")
                                card.data.tags.append("decensored")
                                card.data.tags.append("abliterated")
                                card.text = (
                                    get_readme_intro(
                                        settings,
                                        trial,
                                        evaluator.base_refusals,
                                        evaluator.bad_prompts,
                                    )
                                    + card.text
                                )
                                card.push_to_hub(repo_id, token=token)

                            print(f"Model uploaded to [bold]{repo_id}[/].")

                        case "Chat with the model":
                            print()
                            print(
                                "[cyan]Press Ctrl+C at any time to return to the menu.[/]"
                            )

                            chat = [
                                {"role": "system", "content": settings.system_prompt},
                            ]

                            while True:
                                try:
                                    message = prompt_text(
                                        "User:",
                                        qmark=">",
                                        unsafe=True,
                                    )
                                    if not message:
                                        break
                                    chat.append({"role": "user", "content": message})

                                    print("[bold]Assistant:[/] ", end="")
                                    response = model.stream_chat_response(chat)
                                    chat.append(
                                        {"role": "assistant", "content": response}
                                    )
                                except (KeyboardInterrupt, EOFError):
                                    # Ctrl+C/Ctrl+D
                                    break

                except Exception as error:
                    print(f"[red]Error: {error}[/]")


def main():
    # Install Rich traceback handler.
    install()

    try:
        run()
    except BaseException as error:
        # Transformers appears to handle KeyboardInterrupt (or BaseException)
        # internally in some places, which can re-raise a different error in the handler,
        # masking the root cause. We therefore check both the error itself and its context.
        if isinstance(error, KeyboardInterrupt) or isinstance(
            error.__context__, KeyboardInterrupt
        ):
            print()
            print("[red]Shutting down...[/]")
        else:
            raise
