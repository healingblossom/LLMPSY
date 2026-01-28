# mental_flan.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class MentalFlanWrapper():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("NEU-HAI/mental-flan-t5-xxl")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("NEU-HAI/mental-flan-t5-xxl", device_map="auto", dtype="auto")


    def generate_from_prompt(self, prompt):
        prompt = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**prompt, max_new_tokens=40)

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================================
# Tests
# ============================================================================

    def generate_from_prompt_test(self, prompt):
        return prompt