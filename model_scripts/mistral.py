# mistral.py
import os

from model_manager import model
from prompts.prompt_formatter import build_messages
from tasks_runner import run_all_tasks

from mistral_inference.transformer import Transformer
        from mistral_inference.generate import generate
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_common.protocol.instruct.messages import UserMessage, SystemMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

class MistralWrapper():
    def __init__(self, model_path):
        self.tokenizer = MistralTokenizer.from_file(
            os.path.join(model_path, "tokenizer.model.v3"))
        self.model = Transformer.from_folder(model_path)

    def generate_from_messages(self, messages):
        prompt = build_messages(hier, fehlt, noch, was)

        completion_request = ChatCompletionRequest(messages=prompt)
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = generate(
            [tokens],
            model,
            max_tokens=256,  # wie im CLI-Beispiel
            temperature=0.0,  # deterministisch für Evaluation
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )

        return self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

if __name__ == "__main__":
    run_all_tasks(
        model_name="mistral",
        selected_tasks="task_1_symptom_detection_and_sectioning",
        selected_variants=None  # Alle Varianten
    )



