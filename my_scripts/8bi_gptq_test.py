from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import logging

# quantized_model_dir = "llama-3-8b-instruct-4bit"
# tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=True)
# model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")
# prompt = "What is the capital of France?"
# inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
# output = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(output[0], skip_special_tokens=True))
# pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
# print(pipeline(prompt)[0]["generated_text"])

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

quantized_model_dir = "llama-3-8b-instruct-4bit-128g"
# quantized_model_dir = "llama-3-8b-instruct-2bit-128g"
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")
def generate_response(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
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
