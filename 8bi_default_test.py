from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

model_dir = "Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_dir).to("cuda:0")

def generate_response(prompt, max_new_tokens=50, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature, 
        top_p=top_p, 
        pad_token_id=tokenizer.eos_token_id
    )
    print(f"Prompt: {prompt}")
    print(f"Response: {tokenizer.decode(output[0], skip_special_tokens=True)}\n")

prompts = [
    "What is the capital of Italy?",
    "What is the capital of Portugal?",
    "What is the capital of India?",
    "What is the capital of China?"
]

for prompt in prompts:
    generate_response(prompt)
