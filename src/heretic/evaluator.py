# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>
# THIS IS src/heretic/evaluator.py

import re
from tqdm import tqdm
from datetime import datetime

import pickle
import torch.nn.functional as F
from torch import Tensor

from .config import Settings
from .model import Model
from .utils import Prompt, load_prompts, print


class Evaluator:
    settings: Settings
    model: Model
    good_prompts: list[Prompt]
    bad_prompts: list[Prompt]
    base_logprobs: Tensor
    base_refusals: int

    def __init__(self, settings: Settings, model: Model):
        self.settings = settings
        self.model = model
        pattern = r'\b(' + '|'.join(
            re.escape(marker.lower()) 
            for marker in settings.refusal_markers
        ) + r')\b'
        self.refusal_regex = re.compile(pattern)
        self.has_recorded_initial_responses = False
        self.load_prior = False



        print()
        print(
            f"Loading good evaluation prompts from [bold]{settings.good_evaluation_prompts.dataset} at {datetime.now()}[/]..."
        )
        self.good_prompts = load_prompts(settings, settings.good_evaluation_prompts)
        print(f"* [bold]{len(self.good_prompts)}[/] prompts loaded at {datetime.now()}")

        print(f"* Obtaining first-token probability distributions at {datetime.now()}...")
        self.base_logprobs = model.get_logprobs_batched(self.good_prompts)

        print()
        print(
            f"Loading bad evaluation prompts from [bold]{settings.bad_evaluation_prompts.dataset} at {datetime.now()}[/]..."
        )
        self.bad_prompts = load_prompts(settings, settings.bad_evaluation_prompts)
        print(f"* [bold]{len(self.bad_prompts)}[/] prompts loaded at {datetime.now()}")

        print(f"* Counting model refusals at {datetime.now()}...")
        self.base_refusals = self.count_refusals()
        print(
            f"* Initial refusals: [bold]{self.base_refusals}[/]/{len(self.bad_prompts)} at {datetime.now()}"
        )

    def is_refusal(self, response: str) -> bool:
        # Classify empty responses as refusals to avoid optimizing for them.
        if not response.strip():
            return True

        # Remove emphasis (e.g. "I *will not*...").
        #response = response.lower().replace("*", "")
        #
        ## Normalize typographic apostrophes ("won’t" -> "won't").
        #response = response.replace("’", "'")
        #
        ## Normalize whitespace between words to a single space.
        #response = " ".join(response.split())

        response = " ".join(
            response.lower()
            .replace("*", "")
            .replace("’", "'")
            .split()
        )

        return bool(self.refusal_regex.search(response))
        
        #for marker in self.settings.refusal_markers:
        #    if marker.lower() in response:
        #        return True
        #
        #return False

    def count_refusals(self) -> int:
        refusal_count = 0
        if not self.has_recorded_initial_responses:
            if not self.load_prior:
                print("[INFO] (We need to generate model responses first)")
                responses = self.model.get_responses_batched(
                    self.bad_prompts,
                    skip_special_tokens=True,
                )
                print(f"Finished generating responses at {datetime.now()}")
                
                print(f"Saving responses...")
                with open('dumps/responses.pkl', 'wb') as file:
                    pickle.dump(responses, file)
                print(f"[INFO] Responses have been saved to dumps/responses.pkl at {datetime.now()}; Comment out pickle dumping in src/heretic/evaluator.py and uncomment pickle loading if restarting/resuming.")
                
            else:
                print(f"Loading responses at {datetime.now()}...")
                with open('dumps/responses.pkl', 'rb') as file:
                    responses = pickle.load(file)
                print(f"Finished loading responses at {datetime.now()}...")
            self.has_recorded_initial_responses = True
        
        else:
            print("[INFO] (We need to generate model responses first)")
            responses = self.model.get_responses_batched(
                self.bad_prompts,
                skip_special_tokens=True,
            )
            print(f"Finished generating responses at {datetime.now()}")

        for prompt, response in tqdm(zip(self.bad_prompts, responses), total=len(self.bad_prompts), desc="Counting Refusals"):
            is_refusal = self.is_refusal(response)
            if is_refusal:
                refusal_count += 1

            if self.settings.print_responses:
                print()
                print(f"[bold]System prompt:[/] {prompt.system}")
                print(f"[bold]Prompt:[/] {prompt.user}")
                if not response.strip():
                    response = "[italic]\\[empty][/]"
                print(
                    f"[bold]Response:[/] [{'red' if is_refusal else 'green'}]{response}[/]"
                )

        if self.settings.print_responses:
            print()

        return refusal_count

    def get_score(self) -> tuple[tuple[float, float], float, int]:
        print(f"  * Obtaining first-token probability distributions at {datetime.now()}...")
        logprobs = self.model.get_logprobs_batched(self.good_prompts)
        kl_divergence = F.kl_div(
            logprobs,
            self.base_logprobs,
            reduction="batchmean",
            log_target=True,
        ).item()
        print(f"  * KL divergence: [bold]{kl_divergence:.4f}[/]")

        print("  * Counting model refusals...")
        refusals = self.count_refusals()
        print(f"  * Refusals: [bold]{refusals}[/]/{len(self.bad_prompts)}")

        kl_divergence_scale = self.settings.kl_divergence_scale
        kl_divergence_target = self.settings.kl_divergence_target

        refusals_score = refusals / self.base_refusals

        if kl_divergence >= kl_divergence_target:
            kld_score = kl_divergence / kl_divergence_scale
        else:
            kld_score = refusals_score * kl_divergence_target / kl_divergence_scale

        score = (
            kld_score,
            refusals_score,
        )

        return score, kl_divergence, refusals
