# mental_alpaca.py

from prompts.prompt_builder import AlpacaPromptBuilder
from run_in_env import run_in_conda_env

import tasks_runner

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class MentalAlpacaWrapper():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("NEU-HAI/Llama-2-7b-alpaca-cleaned", use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained( "NEU-HAI/Llama-2-7b-alpaca-cleaned", load_in_8bit=True, device_map="auto", torch_dtype=torch.float16)

    def generate_from_prompt(self, prompt):
        prompt = prompt

        self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                do_sample=False,  # Use greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================================
# Tests
# ============================================================================

    def generate_from_prompt_test(self, prompt):
        return prompt