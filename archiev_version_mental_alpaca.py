from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading model and tokenizer with 8-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained("NEU-HAI/Llama-2-7b-alpaca-cleaned", use_fast=False)
tokenizer.pad_token = tokenizer.unk_token  # Fix padding token issue

# Load model with 8-bit quantization to reduce memory usage
model = AutoModelForCausalLM.from_pretrained(
    "NEU-HAI/Llama-2-7b-alpaca-cleaned",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

print(f"Model loaded with 8-bit quantization on GPU")

# Define the prompt - using Alpaca instruction format
prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
You are a mental health assistant. Analyze the following text and provide supportive guidance.

### Input:
I feel constantly anxious and can't focus on my work lately.

### Response:"""

"""Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\nGenerating response...")
# Generate response - try greedy decoding first for stability
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=150,
        do_sample=False,  # Use greedy decoding
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode and print the response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*50)
print("RESPONSE:")
print("="*50)
print(response)
print("="*50)
