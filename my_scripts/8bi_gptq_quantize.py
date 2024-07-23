from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging
import os

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_dir_base = "llama-3-8b-instruct"

quantize_config = BaseQuantizeConfig(
    bits=2,  
    group_size=128,  
    desc_act=False 
)

quantized_model_dir = f"{quantized_model_dir_base}-{quantize_config.bits}bit-{quantize_config.group_size}g"
os.makedirs(quantized_model_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True, use_auth_token=True)
examples = [tokenizer("This is an example input for quantization.", return_tensors="pt")]

model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config=quantize_config, use_auth_token=True)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
tokenizer.save_pretrained(quantized_model_dir)

print(f"Quantized model saved in directory: {quantized_model_dir}")
