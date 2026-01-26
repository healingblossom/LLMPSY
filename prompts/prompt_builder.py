"""
Prompt generation system for multi-LLM study.
Loads task/snippet configs and generates model-specific prompts.
"""

import yaml
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from itertools import product
from dataclasses import dataclass

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PromptVariant:
    """Represents a single generated prompt variant (model-agnostic)."""
    task_id: str
    episode_type: Optional[str]  # None, "depression", "mania"
    role_type: str  # "role", "none"
    examples_type: str  # "zeroshot", "oneshot", "fewshot"
    instruction: str
    input_template: str

    def format_name(self) -> str:
        """Generate unique, descriptive identifier."""
        parts = [
            self.task_id,
            self.episode_type or "all_episodes",
            f"role_{self.role_type}",
            f"examples_{self.examples_type}"
        ]
        return "_".join(parts)


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class PromptBuilder():
    """
    Abstract base for model-specific prompt builders.

    Each LLM format has specific structural requirements.
    Subclasses implement format-specific formatting while inheriting
    snippet assembly logic from base class.
    """

    def __init__(self):
        """
        Initialize builder with task configuration.

        """
        tasks_config_path = "/home/blossom/PycharmProjects/LLMPSY/config/tasks.yaml" #todo

        with open(tasks_config_path, 'r', encoding='utf-8') as f:
            self.tasks_config = yaml.safe_load(f)

        # Import snippet dictionaries
        from prompts.task_prompts import (
            ROLES, COT, TASKS, FORMATS, INPUTS, EXAMPLES, SYMPTOMS, CRITERIA
        )

        self.roles = ROLES
        self.cot = COT
        self.tasks = TASKS
        self.symptoms = SYMPTOMS
        self.criteria = CRITERIA
        self.formats = FORMATS
        self.inputs = INPUTS
        self.examples = EXAMPLES

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Name of this format (e.g., 'alpaca', 'flan', 'openrouter')."""
        pass

    @abstractmethod
    def format_prompt(self, variant: PromptVariant) -> str:
        """
        Format a PromptVariant into model-specific structure.

        Args:
            variant: PromptVariant to format

        Returns:
            Model-specific formatted prompt string
        """
        pass

    # ========================================================================
    # SHARED SNIPPET ASSEMBLY LOGIC (inherited by all subclasses)
    # ========================================================================

    def _get_snippet(self, collection: Dict[str, str], key: str) -> str:
        """
        Safely retrieve a snippet from a collection.

        Args:
            collection: Dict of snippets (e.g., ROLES, EXAMPLES)
            key: Key to retrieve

        Returns:
            Snippet text or empty string if not found
        """
        return collection.get(key, "")

    def _get_task_config(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task configuration from tasks.yaml."""
        for task in self.tasks_config.get("tasks", []):
            if task.get("task_id") == task_id:
                return task
        return None

    def _build_instruction(
            self,
            task_id: str,
            role_type: str,
            episode_variant: Optional[str] = None,
            cot_active: bool = False
    ) -> str:
        """
        Assemble instruction from role + (cot) + task + format + symptoms.

        Format-Key wird direkt aus tasks.yaml gelesen.
        """
        task_config = self._get_task_config(task_id)
        if not task_config:
            raise ValueError(f"Task {task_id} not found in tasks.yaml")

        snippet_config = task_config.get("snippet_config", {})

        # Get components from config
        role = self._get_snippet(self.roles, role_type)
        task_key = snippet_config.get("task", [])[0] if snippet_config.get("task") else None
        task_text = self._get_snippet(self.tasks, task_key) if task_key else ""
        format_key = snippet_config.get("output_format", [])[0] if snippet_config.get("output_format") else None
        format_text = self._get_snippet(self.formats, format_key) if format_key else ""

        # Substitute episode-dependend lists if needed
        if snippet_config.get("episodes_divided", False) and episode_variant:
            # Für SYMPTOMS (in FORMAT_SYMPTOMS_JSON und FORMAT_SUMMARY)
            if "{symptom}" in format_text:
                symptom_list = self._get_snippet(self.symptoms, episode_variant)
                format_text = format_text.replace("{symptom}", symptom_list)

            # Für CRITERIA (in FORMAT_DIAGNOSTIC_CRITERIA)
            if "{criteria}" in format_text:
                criteria_list = self._get_snippet(self.criteria, episode_variant)  # ← NEUER ZUGRIFF
                format_text = format_text.replace("{criteria}", criteria_list)

        # Combine components
        cot_text = ""
        if snippet_config.get("includes_cot", False):
            cot_text = self._get_snippet(self.cot, "cot")

        instruction = " ".join(filter(None, [role, cot_text, task_text, format_text]))
        return instruction.strip()

    def _build_input_template(self, task_id: str) -> str:
        """
        Build input section based on task config.

        Input-Key(s) werden direkt aus tasks.yaml gelesen.

        Args:
            task_id: Which task to build input for

        Returns:
            Input template with {placeholders} for actual data
        """
        task_config = self._get_task_config(task_id)
        if not task_config:
            raise ValueError(f"Task {task_id} not found in tasks.yaml")

        snippet_config = task_config.get("snippet_config", {})

        # Get input sources from config (is a list)
        input_sources = snippet_config.get("inputs", [])

        if not input_sources:
            raise ValueError(f"No input sources defined for task {task_id}")

        # Combine all input sources
        input_parts = [
            self._get_snippet(self.inputs, source)
            for source in input_sources
        ]
        return "\n\n".join(input_parts)

    def generate_variants_for_task(
            self,
            task_id: str
    ) -> Dict[str, PromptVariant]:
        """
        Generate all prompt permutations for a single task.

        Variant lists come from tasks.yaml config.
        CoT ist keine experimentelle Variable - wird immer verwendet wenn includes_cot: true.

        Args:
            task_id: Task to generate prompts for

        Returns:
            Dict[variant_name -> PromptVariant]
        """
        task_config = self._get_task_config(task_id)
        if not task_config:
            raise ValueError(f"Task {task_id} not found in tasks.yaml")

        snippet_config = task_config.get("snippet_config", {})

        # Use provided variants, or fall back to config
        role_variants = snippet_config.get("role", ["none"])
        examples_variants = snippet_config.get("examples", ["zeroshot"])

        if snippet_config.get("episodes_divided", False):
            episode_variants = snippet_config.get("episode_variants", [None])
        else:
            episode_variants = [None]

        variants = {}

        print(episode_variants)
        print(examples_variants)
        print(role_variants)

        for episode_type, role_type, examples_type in product(
                episode_variants,
                role_variants,
                examples_variants
        ):
            instruction = self._build_instruction(
                task_id=task_id,
                role_type=role_type,
                episode_variant=episode_type
            )

            input_template = self._build_input_template(task_id)

            variant = PromptVariant(
                task_id=task_id,
                episode_type=episode_type,
                role_type=role_type,
                examples_type=examples_type,
                instruction=instruction,
                input_template=input_template
            )

            # Add examples if not zeroshot
            if examples_type != "zeroshot":
                examples_text = self._get_snippet(self.examples, examples_type)
                variant.input_template = (
                        examples_text + "\n\n" + variant.input_template
                )

            variants[variant.format_name()] = variant

        return variants

    def build_all_prompts(self) -> Dict[str, Dict[str, str]]:
        """
        Generate ALL prompts for ALL tasks in this format.

        Returns:
            Nested dict: {task_id -> {variant_name -> formatted_prompt}}  # ← formatierte Strings!
        """
        all_prompts = {}

        for task in self.tasks_config.get("tasks", []):
            task_id = task.get("task_id")
            print(f"[{self.format_name.upper()}] Generating prompts for {task_id}...")

            variants = self.generate_variants_for_task(task_id)

            all_prompts[task_id] = {
                name: self.format_prompt(variant)  # format_prompt(PromptVariant) → String
                for name, variant in variants.items()
            }

        return all_prompts

    def save_prompts(
            self,
            prompts: Dict[str, Dict[str, str]],
            output_dir: str = "./prompts/",
            output_format: str = "json"
    ) -> None:
        """
        Save generated prompts to files.

        Args:
            prompts: Output from build_all_prompts()
            output_dir: Directory to save to
            output_format: "yaml" or "json"
        """
        import os

        # Create format-specific subdirectory
        format_dir = os.path.join(output_dir, f"{self.format_name}/")
        os.makedirs(format_dir, exist_ok=True)

        for task_id, variants in prompts.items():
            output_file = os.path.join(format_dir, f"{task_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(variants, f, indent=2, ensure_ascii=False)
            print(f"✓ [{self.format_name}] Saved: {output_file}")

    def get_formated_prompts(self, model_name):

        model_spec = self.get_model_spec(model_name)
        format = model_spec['format']

        print(f"   Format: {format}")

        if format == 'alpaca':
            return AlpacaPromptBuilder.build_all_prompts()

        elif format == 'mistral':
            return MistralPromptBuilder.build_all_prompts()

        elif format == 'flan':
            return FLANPromptBuilder.build_all_prompts()

        elif format == 'openrouter':
            return OpenRouterPromptBuilder.build_all_prompts()

        else:
            raise NotImplementedError(f"Source '{format}' nicht implementiert")


# ============================================================================
# FORMAT IMPLEMENTATIONS
# ============================================================================
class FLANPromptBuilder(PromptBuilder):
    """
    FLAN format builder.

    FLAN uses task-specific prefixes and instruction-tuning.
    Format is minimal - just instruction followed by input.

    Example from Google FLAN:
    "Question: {question}\nAnswer:"
    """

    @property
    def format_name(self) -> str:
        return "flan"

    def format_prompt(self, variant: PromptVariant) -> str:
        """
        Format PromptVariant into FLAN structure.

        FLAN models typically use minimal formatting:
        Instruction/Context on first line(s), then the input/question.
        """
        # FLAN doesn't use explicit structure markers
        prompt = f"""{variant.instruction}

{variant.input_template}"""
        return prompt.strip()


class OpenRouterPromptBuilder(PromptBuilder):
    """
    OpenRouter format builder (Chat Completions API).

    OpenRouter standardizes across 400+ models using OpenAI API format.
    Returns prompt as dict with 'messages' for API consumption.

    Format:
    ```json
    {
      "messages": [
        {"role": "system", "content": "...instruction..."},
        {"role": "user", "content": "...input..."}
      ]
    }
    ```
    """

    @property
    def format_name(self) -> str:
        return "openrouter"

    def format_prompt(self, variant: PromptVariant) -> str:
        """
        Format PromptVariant into OpenRouter Messages API structure.

        Returns JSON string for API consumption.
        """
        messages = [
            {
                "role": "system",
                "content": variant.instruction
            },
            {
                "role": "user",
                "content": variant.input_template
            }
        ]

        return json.dumps({"messages": messages}, indent=2, ensure_ascii=False)


class MistralPromptBuilder(PromptBuilder):
    """
    Mistral format builder.

    Mistral-7B-Instruct uses simple instruction/response structure.
    Format:
    ```
    [INST] {instruction}

    {input} [/INST]
    ```
    """

    @property
    def format_name(self) -> str:
        return "mistral"

    def format_prompt(self, variant: PromptVariant) -> str:
        """Format PromptVariant into Mistral structure."""
        template = """[INST] {instruction}

{input_template} [/INST]"""

        return template.format(
            instruction=variant.instruction,
            input_template=variant.input_template
        )


class AlpacaPromptBuilder(PromptBuilder):
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

if __name__ == "__main__":
    alpaca_builder = AlpacaPromptBuilder()
    all_prompts = alpaca_builder.build_all_prompts()

    for task_id, variants in all_prompts.items():
        print(f"\n  Task: {task_id}")
        print(f"    Variants: {len(variants)}")
        for variant_name, formatted_prompt in list(variants.items())[:1]:  # Nur erste zeigen
            print(f"      - {variant_name}")
            print(f"        Prompt :\n{formatted_prompt}...")