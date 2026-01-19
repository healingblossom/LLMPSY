# mental_alpaca.py
from model_manager import model
from prompts.prompt_builder  import ModelFormatBuilder, PromptVariant
from tasks_runner import run_all_tasks

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

class AlpacaPromptBuilder(ModelFormatBuilder):
    """
    Alpaca format builder.
    https://github.com/tatsu-lab/alpaca

    Format:
    ```
    Below is an instruction that describes a task, paired with an input...
    ### Instruction: ...
    ### Input: ...
    ### Response:
    ```
    """

    @property
    def format_name(self) -> str:
        return "alpaca"

    def format_prompt(self, variant: PromptVariant) -> str:
        """Format PromptVariant into Alpaca structure."""
        template = """\
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_template}

### Response:
"""
        return template.format(
            instruction=variant.instruction,
            input_template=variant.input_template
        )


# ============================================================================
# Tests
# ============================================================================

if __name__ == "__main__":
    run_all_tasks(
        model_name="mental_alpaca",
        selected_tasks="task_1_symptom_detection_and_sectioning",
        selected_variants=None  # Alle Varianten
    )

    alpaca_builder = AlpacaPromptBuilder()
    all_prompts = alpaca_builder.build_all_prompts()

    for task_id, variants in all_prompts.items():
        print(f"\n  Task: {task_id}")
        print(f"    Variants: {len(variants)}")
        for variant_name, formatted_prompt in list(variants.items())[:1]:  # Nur erste zeigen
            print(f"      - {variant_name}")
            print(f"        Prompt :\n{formatted_prompt}...")

