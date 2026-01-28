# mental_alpaca.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class MentalAlpacaWrapper():
    def __init__(self):
        local_dir = "/sc-scratch/sc-scratch-dhzc-psycho/data/models/Llama-2-7b-alpaca-cleaned"  # todo von config nehmen
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir, device_map="auto", dtype="auto")
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained(local_dir, device_map="auto", dtype="auto")

    def generate_from_prompt(self, prompt):
        prompt = self.tokenizer(prompt, return_tensors="pt").to("cuda")

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
    model = MentalAlpacaWrapper()
    print("3")
    answer = model.generate_from_prompt("hi, antworte mit 'nein'")
    print(answer)


if __name__ == "__main__":
    main()

