# mistral_wrapper.py

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

    def generate_from_messages(self, prompt):
        prompt = prompt

        completion_request = ChatCompletionRequest(messages=prompt)
        tokens = self.tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = generate(
            [tokens],
            self.model,
            max_tokens=256,  # wie im CLI-Beispiel
            temperature=0.0,  # deterministisch für Evaluation
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id,
        )

        return self.tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

# ============================================================================
# Tests
# ============================================================================

    def generate_from_prompt_test(self, prompt):
        return prompt



