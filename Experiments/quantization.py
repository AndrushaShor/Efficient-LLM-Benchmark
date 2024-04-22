from transformers import BitsAndBytesConfig 
import torch

'''
Add additional configs here
'''

# Quantization
CONFIG_4BITS = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) # For QLORA
CONFIG_4BITS_NORM = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, "float16")) # For QLORA and GEMMA
CONFIG_4BITS_NORM_NESTED = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, "float16"), bnb_4bit_use_double_quant=True) # For QLORA and GEMMA
CONFIG_4BITS_NESTED = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True) # For QLORA
CONFIG_8BITS = BitsAndBytesConfig(load_in_8bit=True)


