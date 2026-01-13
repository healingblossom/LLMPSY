# mental_alpaca.py
from model_manager import model
from prompts.prompt_formatter import build_messages
from tasks_runner import run_all_tasks

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class MentalAlpacaWrapper():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("NEU-HAI/Llama-2-7b-alpaca-cleaned", use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained( "NEU-HAI/Llama-2-7b-alpaca-cleaned", load_in_8bit=True, device_map="auto", torch_dtype=torch.float16)

    def generate_from_messages(self, messages):
        prompt = build_messages(hier, fehlt, noch, was)

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

if __name__ == "__main__":
    run_all_tasks(
        model_name="mental_alpaca",
        selected_tasks="task_1_symptom_detection_and_sectioning",
        selected_variants=None  # Alle Varianten
    )
