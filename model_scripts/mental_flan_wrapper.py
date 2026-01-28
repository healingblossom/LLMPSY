# mental_flan.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pathlib import Path
import torch


class MentalFlanWrapper():
    def __init__(self):
        local_dir = "/sc-scratch/sc-scratch-dhzc-psycho/data/models/mental-flan-t5-xxl"  # todo von config nehmen

        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(local_dir, device_map="auto", dtype="auto",
                                                           local_files_only=True)

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


def main():
    print("1")
    model = MentalFlanWrapper()
    print("3")
    answer = model.generate_from_prompt("hi, antworte mit 'nein'")
    print(answer)


if __name__ == "__main__":
    main()
