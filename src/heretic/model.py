# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>
# THIS IS src/heretic/model.py


import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import Linear
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import Module, ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,  # ty:ignore[possibly-missing-import]
)

from .config import QuantizationMethod, Settings
from .utils import Prompt, batchify, empty_cache, print



from tqdm import tqdm
from transformers import GPTQConfig
try:
    from auto_gptq import AutoGPTQForCausalLM
    HAS_AUTO_GPTQ = True
except ImportError:
    HAS_AUTO_GPTQ = False
try:
    from gptqmodel import GPTQModel
    HAS_GPTQMODEL = True
except ImportError:
    HAS_GPTQMODEL = False
@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float
    
    def to_dict(self) -> dict:
        return {
            "max_weight": self.max_weight,
            "max_weight_position": self.max_weight_position,
            "min_weight": self.min_weight,
            "min_weight_distance": self.min_weight_distance,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AbliterationParameters":
        return cls(
            max_weight=data["max_weight"],
            max_weight_position=data["max_weight_position"],
            min_weight=data["min_weight"],
            min_weight_distance=data["min_weight_distance"],
        )


class Model:
    model: PreTrainedModel | PeftModel
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, settings: Settings):
        self.settings = settings
        self.response_prefix = ""
        self.needs_reload = False

        print()
        print(f"Loading model [bold]{settings.model}[/]...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.model,
            trust_remote_code=settings.trust_remote_code,
        )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CRITICAL: Always use left-padding for decoder-only models during generation.
        #           Right-padding causes empty outputs because the model sees PAD tokens
        #           after the prompt and thinks the sequence is complete.
        self.tokenizer.padding_side = "left"

        self.model = None  # ty:ignore[invalid-assignment]
        self.max_memory = (
            {int(k) if k.isdigit() else k: v for k, v in settings.max_memory.items()}
            if settings.max_memory
            else None
        )
        self.trusted_models = {settings.model: settings.trust_remote_code}

        if self.settings.evaluate_model is not None:
            self.trusted_models[settings.evaluate_model] = settings.trust_remote_code

        if self.settings.quantization == QuantizationMethod.GPTQ or self.settings.quantization == QuantizationMethod.AGPTQ:
            try:
                try:
                    from gptqmodel import GPTQModel
                    HAS_GPTQMODEL = True
                    print("gptqmodel installed")
                except ImportError:
                    HAS_GPTQMODEL = False
                    print("gptqmodel NOT installed")
                if self.settings.quantization == QuantizationMethod.GPTQ and HAS_GPTQMODEL:
                    print(f"* Loading GPTQ model...")
                    self.model = GPTQModel.load(
                        settings.model,
                        device_map=settings.device_map,
                        max_memory=self.max_memory,
                        trust_remote_code=self.trusted_models.get(settings.model),
                        disable_exllama=False,  # Use exllama kernels if available
                        use_triton=False,       # Disable Triton (Windows compatibility)
                    )
                    print("Successfully loaded GPTQ model")
                if not HAS_GPTQMODEL or (self.settings.quantization == QuantizationMethod.AGPTQ and HAS_AUTO_GPTQ):
                    print(f"* Loading GPTQ model ({self.settings.gptq_bits}-bit) via Auto-GPTQ...")
                    self.model = AutoGPTQForCausalLM.from_quantized(
                        settings.model,
                        device_map=settings.device_map,
                        max_memory=self.max_memory,
                        trust_remote_code=self.trusted_models.get(settings.model),
                        use_safetensors=True,
                        disable_exllama=False,
                        use_triton=False,
                    )
                    print("Successfully loaded Auto-GPTQ model")
                
                if (not HAS_GPTQMODEL and not HAS_AUTO_GPTQ) or self.model == None:
                    print("NEITHER!!!!")
                    raise ImportError("Neither gptqmodel nor auto-gptq is installed")
                    
                
                self.generate(
                    [
                        Prompt(
                            system=settings.system_prompt,
                            user="What is 1+1?",
                        )
                    ],
                    max_new_tokens=1,
                )
                
                print("[green]Ok[/]")
                print(f"[bold green]Model loaded (GPTQ).[/]")
            except Exception as error:
                self.model = None  # ty:ignore[invalid-assignment]
                empty_cache()
                print(f"[red]Failed to load GPTQ model[/] ({error})")
                raise Exception(f"Failed to load GPTQ model: {error}")
        else:
            for dtype in settings.dtypes:
                print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

                try:
                    quantization_config = self._get_quantization_config(dtype)

                    extra_kwargs = {}
                    if quantization_config is not None:
                        extra_kwargs["quantization_config"] = quantization_config

                    self.model = AutoModelForCausalLM.from_pretrained(
                        settings.model,
                        torch_dtype=dtype,
                        device_map=settings.device_map,
                        max_memory=self.max_memory,
                        trust_remote_code=self.trusted_models.get(settings.model),
                        attn_implementation="flash_attention_2",
                        **extra_kwargs,
                    )

                    if self.trusted_models.get(settings.model) is None:
                        self.trusted_models[settings.model] = True

                    # Test run
                    self.generate(
                        [
                            Prompt(
                                system=settings.system_prompt,
                                user="What is 1+1?",
                            )
                        ],
                        max_new_tokens=1,
                    )
                except Exception as error:
                    self.model = None
                    empty_cache()
                    print(f"[red]Failed[/] ({error})")
                    continue

                print("[green]Ok[/]")
                if settings.quantization == QuantizationMethod.BNB_4BIT:
                    print("[bold green]Model loaded in 4-bit precision (BitsAndBytes).[/]")
                break

            if self.model is None:
                raise Exception("Failed to load model with all configured dtypes.")
        
        self._apply_lora()

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component, modules in self.get_layer_modules(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(modules)}[/] modules per layer"
            )

    def _apply_lora(self):
        is_qmodel = type(self.model).__name__.endswith('QModel')
        
        if is_qmodel:
            # Get the INNER model (MistralForCausalLM), NOT the QModel wrapper
            inner_model = self.model._modules['model']
            
            # Wrap ONLY the inner model with PEFT
            target_modules = [comp.split(".")[-1] for comp in self._get_abliterable_components_from_model(inner_model)]
            
            peft_config = LoraConfig(
                r=1,
                target_modules=target_modules,
                lora_alpha=1,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Wrap the inner model, replace it in the QModel
            peft_model = get_peft_model(inner_model, peft_config)
            self.model._modules['model'] = peft_model
        else:
            # Non-GPTQ path - wrap self.model directly
            target_modules = [comp.split(".")[-1] for comp in self.get_abliterable_components()]
            
            peft_config = LoraConfig(
                r=1,
                target_modules=target_modules,
                lora_alpha=1,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            self.model = get_peft_model(self.model, peft_config)
        
        print(f"[green]LoRA adapters initialized[/]")


    def _get_abliterable_components_from_model(self, model) -> list[str]:
        """Get abliterable components from an unwrapped model."""
        # Find layers in the model
        modules = model._modules
        if 'model' in modules and 'layers' in modules['model']._modules:
            layer = modules['model']._modules['layers']._modules['0']
        elif 'layers' in modules:
            layer = modules['layers']._modules['0']
        else:
            raise AttributeError("Cannot find layers")
        
        return ['attn.o_proj', 'mlp.down_proj']  # Or detect dynamically

    def _get_quantization_config(self, dtype: str) -> BitsAndBytesConfig | GPTQConfig | None:
        """
        Creates quantization config based on settings.

        Args:
            dtype: The dtype string (e.g., "auto", "bfloat16")

        Returns:
            BitsAndBytesConfig, GPTQConfig, or None
        """
        if self.settings.quantization == QuantizationMethod.BNB_4BIT:
            # BitsAndBytesConfig expects a torch.dtype, not a string.
            if dtype == "auto":
                compute_dtype = torch.bfloat16
            else:
                compute_dtype = getattr(torch, dtype)

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        elif self.settings.quantization == QuantizationMethod.AGPTQ:
            # Validate bits
            if self.settings.gptq_bits not in [2, 3, 4, 8]:
                raise ValueError(
                    f"GPTQ bits must be 2, 3, 4, or 8. Got: {self.settings.gptq_bits}"
                )
            
            return GPTQConfig(
                bits=self.settings.gptq_bits,
                group_size=self.settings.gptq_group_size,
                desc_act=self.settings.gptq_desc_act,
                disable_exllama=True,  # Better compatibility with LoRA
            )
        
        return None

    def get_merged_model(self) -> PreTrainedModel:
        # Get the actual PEFT model (may be wrapped in QModel)
        peft_model = self.model
        is_qmodel = type(peft_model).__name__.endswith('QModel')
        
        if is_qmodel:
            peft_model = peft_model.model
            
        # Guard against calling this method at the wrong time.
        if not isinstance(peft_model, PeftModel):
            raise TypeError(f"Expected PeftModel, got {type(peft_model)}")

        # Check if we need special handling for quantized models
        if self.settings.quantization in [QuantizationMethod.BNB_4BIT, QuantizationMethod.GPTQ, QuantizationMethod.AGPTQ]:
            # Quantized models need special handling - we must reload the base model
            # in full precision to merge the LoRA adapters

            # Get the adapter state dict before we do anything
            adapter_state = {}
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    adapter_state[name] = param.data.clone().cpu()

            # For GPTQ, we can try to stay quantized during merge (less memory)
            if self.settings.quantization == QuantizationMethod.GPTQ:
                print("* Merging LoRA adapters (keeping GPTQ quantization)...")
                # GPTQ models can often merge in-place without full dequantization
                try:
                    merged_model = peft_model.merge_and_unload()
                    self.needs_reload = True
                    return merged_model
                except Exception as e:
                    print(f"[yellow]In-place merge failed: {e}[/]")
                    print("[yellow]Falling back to full reload method...[/]")


            # Load base model in full precision on CPU to avoid VRAM issues
            print("* Loading base model on CPU (this may take a while)...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.settings.model,
                torch_dtype=self.model.dtype,
                device_map="cpu",
                trust_remote_code=self.trusted_models.get(self.settings.model),
            )

            # Apply LoRA adapters to the CPU model

            print("* Applying LoRA adapters...")
            target_modules = self.get_abliterable_components()
            peft_config = LoraConfig(
                r=1,
                target_modules=target_modules,
                lora_alpha=1,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            )
            peft_model = get_peft_model(base_model, peft_config)

            # Copy the trained adapter weights
            for name, param in peft_model.named_parameters():
                if name in adapter_state:
                    param.data = adapter_state[name].to(param.device)

            # Merge and unload
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            return merged_model
        else:
            # Non-quantized model - can merge directly
            print("* Merging LoRA adapters into base model...")
            merged_model = peft_model.merge_and_unload()
            # merge_and_unload() modifies self.model in-place, destroying LoRA adapters.
            # Mark for full reload if user switches trials later.
            self.needs_reload = True
            return merged_model

    def reset_model(self):
        """
        Resets the model to a clean state for the next trial or evaluation.

        Behavior:
        - Fast path: If the same model is loaded and doesn't need full reload,
          resets LoRA adapter weights to zero (identity transformation).
        - Slow path: If switching models or after merge_and_unload(),
          performs full model reload with quantization config.
        """
        is_qmodel = type(self.model).__name__.endswith('QModel')
        inner_model = self.model.model if is_qmodel else self.model
        current_model = getattr(inner_model.config if hasattr(inner_model, 'config') else self.model, "name_or_path", None)
       
        if current_model == self.settings.model and not self.needs_reload:
            # Reset LoRA adapters to zero (identity transformation)
            model_to_reset = inner_model
            
            for name, module in model_to_reset.named_modules():
                if "lora_B" in name and hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
            return

        # Purge existing model object from memory to make space.
        self.model = None  # ty:ignore[invalid-assignment]
        empty_cache()

        # Special handling for GPTQ
        if self.settings.quantization == QuantizationMethod.GPTQ or self.settings.quantization.AGPTQ:
            try:
                from gptqmodel import GPTQModel
                HAS_GPTQMODEL = True
            except ImportError:
                HAS_GPTQMODEL = False
            
            if HAS_GPTQMODEL:
                self.model = GPTQModel.load(
                    self.settings.model,
                    device_map=self.settings.device_map,
                    max_memory=self.max_memory,
                    offload_buffers=True,
                    trust_remote_code=self.trusted_models.get(self.settings.model),
                )
            elif HAS_AUTO_GPTQ:
                self.model = AutoGPTQForCausalLM.from_quantized(
                    self.settings.model,
                    device_map=self.settings.device_map,
                    max_memory=self.max_memory,
                    trust_remote_code=self.trusted_models.get(self.settings.model),
                    use_safetensors=True,
                )
            else:
                raise ImportError("Neither gptqmodel nor auto-gptq is installed")
        else:
            # Non-GPTQ models
            dtype = self.model.dtype if hasattr(self, 'model') and self.model else torch.bfloat16
            quantization_config = self._get_quantization_config(str(dtype).split(".")[-1])

            extra_kwargs = {}
            if quantization_config is not None:
                extra_kwargs["quantization_config"] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(
                self.settings.model,
                torch_dtype=dtype,
                device_map=self.settings.device_map,
                max_memory=self.max_memory,
                trust_remote_code=self.trusted_models.get(self.settings.model),
                attn_implementation="flash_attention_2",
                **extra_kwargs,
            )

        self._apply_lora()
        self.needs_reload = False

    def debug_model_structure(self):
        """Print model structure for debugging."""
        def print_modules(obj, prefix="", depth=0, max_depth=4):
            if depth > max_depth:
                return
            print(f"{prefix}Type: {type(obj).__name__}")
            
            # Check _modules dict
            modules = getattr(obj, '_modules', None)
            if modules:
                print(f"{prefix}_modules keys: {list(modules.keys())}")
                for key, val in modules.items():
                    if key in ('layers', 'model', 'base_model', 'language_model'):
                        print(f"{prefix}  [{key}] ->")
                        print_modules(val, prefix + "    ", depth + 1, max_depth)
            
            # Check specific attributes without triggering __getattr__
            for attr in ['model', 'base_model', 'layers']:
                if attr in dir(obj.__class__):  # Only check if it's a real attribute
                    try:
                        val = object.__getattribute__(obj, attr)
                        print(f"{prefix}.{attr} (direct): {type(val).__name__}")
                    except:
                        pass
        
        print("=" * 60)
        print("MODEL STRUCTURE DEBUG:")
        print("=" * 60)
        print_modules(self.model)
        print("=" * 60)


    def get_layers(self) -> ModuleList:
        #print("WE RAN")
        #self.debug_model_structure()
        model = self.model

        # For gptqmodel's quantized models, the structure is:
        # LlamaQModel._modules['model'] -> (possibly PeftModel) -> MistralForCausalLM
        if type(model).__name__.endswith('QModel'):
            if hasattr(model, '_modules') and 'model' in model._modules:
                model = model._modules['model']

        # Unwrap PeftModel (if present after _apply_lora)
        if isinstance(model, PeftModel):
            model = model.base_model.model

        # Try direct _modules access ONLY (avoid __getattr__ recursion)
        def get_via_modules(obj, *keys):
            """Safely traverse _modules dict without triggering __getattr__"""
            for key in keys:
                modules = getattr(obj, '_modules', None)
                if modules is None or key not in modules:
                    return None
                obj = modules[key]
            return obj

        # 1. Try: model._modules['model']._modules['layers']
        layers = get_via_modules(model, 'model', 'layers')
        if isinstance(layers, ModuleList):
            return layers

        # 2. Try: model._modules['layers']
        layers = get_via_modules(model, 'layers')
        if isinstance(layers, ModuleList):
            return layers

        # 3. Try: model._modules['model']._modules['language_model']._modules['layers']
        layers = get_via_modules(model, 'model', 'language_model', 'layers')
        if isinstance(layers, ModuleList):
            return layers

        raise AttributeError(
            f"Could not locate transformer layers. "
            f"Model type: {type(model).__name__}, "
            f"Self.model type: {type(self.model).__name__}"
        )

    def get_layer_modules(self, layer_index: int) -> dict[str, list[Module]]:
        layer = self.get_layers()[layer_index]

        modules = {}

        def try_add(component: str, module: Any):
            # Only add if it's a proper nn.Module (PEFT can wrap these with LoRA)
            if isinstance(module, Module):
                if component not in modules:
                    modules[component] = []
                modules[component].append(module)
            else:
                # Assert for unexpected types (catches architecture changes)
                assert not isinstance(module, Tensor), (
                    f"Unexpected Tensor in {component} - expected nn.Module"
                )

        # Exceptions aren't suppressed here, because there is currently
        # no alternative location for the attention out-projection.
        try_add("attn.o_proj", layer.self_attn.o_proj)  # ty:ignore[possibly-missing-attribute]

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.down_proj)  # ty:ignore[possibly-missing-attribute]

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.w2)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear)  # ty:ignore[possibly-missing-attribute]

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:  # ty:ignore[possibly-missing-attribute, not-iterable]
                try_add("mlp.down_proj", expert.output_linear)  # ty:ignore[possibly-missing-attribute]

        # We need at least one module across all components for abliteration to work.
        total_modules = sum(len(mods) for mods in modules.values())
        assert total_modules > 0, "No abliterable modules found in layer"

        return modules

    
    def debug_quantlinear(self, module):
        """Debug GPTQ TorchQuantLinear structure."""
        print(f"Type: {type(module).__name__}")
        print(f"Attributes: {[a for a in dir(module) if not a.startswith('_')]}")
        
        # Common GPTQ weight storage attributes
        for attr in ['qweight', 'qzeros', 'scales', 'g_idx', 'bias', 'weight']:
            if hasattr(module, attr):
                val = getattr(module, attr)
                if hasattr(val, 'shape'):
                    print(f"  {attr}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"  {attr}: {type(val)}")


    def get_abliterable_components(self) -> list[str]:
        return list(self.get_layer_modules(0).keys())

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        if direction_index is None:
            refusal_direction = None
        else:
            # The index must be shifted by 1 because the first element
            # of refusal_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(len(self.get_layers())):
            for component, modules in self.get_layer_modules(layer_index).items():
                params = parameters[component]

                # Type inference fails here for some reason.
                distance = cast(float, abs(layer_index - params.max_weight_position))

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    # The index must be shifted by 1 because the first element
                    # of refusal_directions is the direction for the embeddings.
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                for module in modules:
                    # FIXME: This cast is potentially invalid, because the program logic
                    #        does not guarantee that the module is of type Linear, and in fact
                    #        the retrieved modules might not conform to the interface assumed
                    #        below (though they do in practice). However, this is difficult
                    #        to fix cleanly, because get_layer_modules is called twice on
                    #        different model configurations, and PEFT employs different
                    #        module types depending on the chosen quantization.
                    module = cast(Linear, module)

                    # LoRA abliteration: delta W = -lambda * v * (v^T W)
                    # lora_B = -lambda * v
                    # lora_A = v^T W

                    # Use the FP32 refusal direction directly (no downcast/upcast)
                    # and move to the correct device.
                    v = layer_refusal_direction.to(module.weight.device)

                    # Get W (dequantize if necessary).
                    #
                    # FIXME: This cast is valid only under the assumption that the original
                    #        module wrapped by the LoRA adapter has a weight attribute.
                    #        See the comment above for why this is currently not guaranteed.
                    #self.debug_quantlinear(module.base_layer)
                    base_layer = module.base_layer

                    # Check if it's a GPTQ quantized layer
                    if hasattr(base_layer, 'dequantize_weight'):
                        # GPTQ quantized layer
                        W = base_layer.dequantize_weight().to(torch.float32)
                    elif hasattr(base_layer, 'weight'):
                        base_weight = base_layer.weight
                        quant_state = getattr(base_weight, "quant_state", None)
                        
                        if quant_state is None:
                            W = base_weight.to(torch.float32)
                        else:
                            # BitsAndBytes 4-bit quantization
                            W = cast(
                                Tensor,
                                bnb.functional.dequantize_4bit(
                                    base_weight.data,
                                    quant_state,
                                ).to(torch.float32),
                            )
                    else:
                        raise AttributeError(
                            f"Cannot get weights from {type(base_layer).__name__}. "
                            f"No 'weight' or 'dequantize_weight' found."
                        )

                    if W.shape[0] != v.shape[0]:
                        W = W.T

                    # Calculate lora_A = v^T W
                    # v is (d_out,), W is (d_out, d_in)
                    # v @ W -> (d_in,)
                    lora_A = (v @ W).view(1, -1)

                    # Calculate lora_B = -weight * v
                    # v is (d_out,)
                    lora_B = (-weight * v).view(-1, 1)

                    # Assign to adapters. The adapter name is "default", because that's
                    # what PEFT uses when no name is explicitly specified, as above.
                    # These casts are therefore valid.
                    weight_A = cast(Tensor, module.lora_A["default"].weight)
                    weight_B = cast(Tensor, module.lora_B["default"].weight)
                    weight_A.data = lora_A.to(weight_A.dtype)
                    weight_B.data = lora_B.to(weight_B.dtype)

    def generate(
        self,
        prompts: list[Prompt],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateDecoderOnlyOutput | LongTensor]:
        chats = [
            [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ]
            for prompt in prompts
        ]

        # This cast is valid because list[str] is the return type
        # for batched operation with tokenize=False.
        chat_prompts = cast(
            list[str],
            self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        if self.response_prefix:
            # Append the common response prefix to the prompts so that evaluation happens
            # at the point where responses start to differ for different prompts.
            chat_prompts = [prompt + self.response_prefix for prompt in chat_prompts]

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )  # ty:ignore[call-non-callable]

        return inputs, outputs

    def get_responses(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        return self.tokenizer.batch_decode(
            # Extract the newly generated part.
            # This cast is valid because the input_ids property is a Tensor
            # if the tokenizer is invoked with return_tensors="pt", as above.
            outputs[:, cast(Tensor, inputs["input_ids"]).shape[1] :],
            skip_special_tokens=skip_special_tokens,
        )

    def get_responses_batched(
        self,
        prompts: list[Prompt],
        skip_special_tokens: bool = False,
    ) -> list[str]:
        responses = []

        batches = batchify(prompts, self.settings.batch_size)
        for batch in tqdm(batches, desc="Generating Response Batch"):
        #for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(
                batch,
                skip_special_tokens=skip_special_tokens,
            ):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Hidden states for the first (only) generated token.
        # This cast is valid because we passed output_hidden_states=True above.
        hidden_states = cast(tuple[tuple[FloatTensor]], outputs.hidden_states)[0]

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        return residuals.to(torch.float32)

    def get_residuals_batched(self, prompts: list[Prompt]) -> Tensor:
        batches = batchify(prompts, self.settings.batch_size)
        first_batch_residuals = self.get_residuals(batches[0])
        n_prompts = len(prompts)
        n_layers, n_components = first_batch_residuals.shape[1], first_batch_residuals.shape[2]
        
        # Pre-allocate on CPU to avoid GPU memory fragmentation
        result = torch.empty(
            (n_prompts, n_layers, n_components),
            dtype=torch.float32,
            device='cpu'
        )
        result[:len(batches[0])] = first_batch_residuals.cpu()
        offset = len(batches[0])
        
        for batch in tqdm(batches[1:], initial=1, total=len(batches), desc="Processing Residuals"):
            batch_residuals = self.get_residuals(batch)
            batch_size = len(batch)
            result[offset:offset + batch_size] = batch_residuals.cpu()
            offset += batch_size

        return result

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[Prompt]) -> Tensor:
        # We only generate one token, and we return the (log) probability distributions
        # over the vocabulary at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # This cast is valid because GenerateDecoderOnlyOutput is the return type
        # of model.generate with return_dict_in_generate=True.
        outputs = cast(GenerateDecoderOnlyOutput, outputs)

        # Logits for the first (only) generated token.
        # This cast is valid because we passed output_scores=True above.
        logits = cast(tuple[FloatTensor], outputs.scores)[0]

        # The returned tensor has shape (prompt, token).
        return F.log_softmax(logits, dim=-1)

    def get_logprobs_batched(self, prompts: list[Prompt]) -> Tensor:
        batches = batchify(prompts, self.settings.batch_size)
        first_batch_logprobs = self.get_logprobs(batches[0])
        n_prompts = len(prompts)
        vocab_size = first_batch_logprobs.shape[1]
        
        # Pre-allocate on CPU to avoid GPU memory fragmentation
        result = torch.empty(
            (n_prompts, vocab_size),
            dtype=torch.float32,
            device='cpu'
        )
        result[:len(batches[0])] = first_batch_logprobs.cpu()
        offset = len(batches[0])
        
        for batch in tqdm(batches[1:], initial=1, total=len(batches), desc="Processing LogProbs"):
            batch_logprobs = self.get_logprobs(batch)
            batch_size = len(batch)
            result[offset:offset + batch_size] = batch_logprobs.cpu()
            offset += batch_size
        return result

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        # This cast is valid because str is the return type
        # for single-chat operation with tokenize=False.
        chat_prompt = cast(
            str,
            self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
            ),
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            # The TextStreamer constructor annotates this parameter with the AutoTokenizer
            # type, which makes no sense because AutoTokenizer is a factory class,
            # not a base class that tokenizers inherit from.
            self.tokenizer,  # ty:ignore[invalid-argument-type]
            skip_prompt=True,
            skip_special_tokens=True,
        )

        # FIXME: The type checker has been disabled here because of the extremely complex
        #        interplay between different generate() signatures and dynamic delegation.
        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )  # ty:ignore[call-non-callable]

        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )