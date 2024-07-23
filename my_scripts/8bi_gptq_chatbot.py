from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

quantized_model_dir = "llama-3-8b-instruct-4bit-128g"
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

def generate_response(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

print("Welcome! Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Ending the conversation. Goodbye!")
        break
    response = generate_response(user_input, 512)
    print(f"Bot: {response}\n")
