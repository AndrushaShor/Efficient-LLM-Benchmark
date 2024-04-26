import os
import gc
import re
import json
import torch
import logging 
import time
import pandas as pd


from collections import defaultdict
from datasets import Dataset
import accelerate
import bitsandbytes

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, IA3Config, AdaLoraConfig, PromptEmbedding, PromptTuningConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from trl import SFTTrainer

'''
Please referenced this github for more info on how PeFT was implemented by the wonderful folks at Huggingface:
https://github.com/huggingface/peft
'''

def load_tokenized_dataset(file_path:str) -> Dataset:
    data_dict = {}
    with open(file_path, 'r') as fp:
        id, questions, answers, text, input_id = json.load(fp)

        data_dict['id'] = id
        data_dict['questions'] = questions
        data_dict['answers'] = answers
        data_dict['text'] = text
        data_dict['input_ids'] = input_id


    return Dataset.from_dict(data_dict)

def load_datasets_from_directory(directory_path: str) -> tuple:
    
    expected_files = {"train.json", "dev.json", "test.json"}
    
    
    actual_files = set(os.listdir(directory_path))
    
    
    if expected_files != actual_files:
        raise ValueError(f"Directory must contain exactly these files: {expected_files}")
    
    
    train_dataset = load_tokenized_dataset(os.path.join(directory_path, "train.json"))
    dev_dataset = load_tokenized_dataset(os.path.join(directory_path, "dev.json"))
    test_dataset = load_tokenized_dataset(os.path.join(directory_path, "test.json"))

    return (train_dataset, dev_dataset, test_dataset)

def load_model(base_model: str, bnb_config:BitsAndBytesConfig=None, access_token:str=None, on_gpu:bool=False, use_cache:bool=False, pretraining_tp:int=1) -> AutoModelForCausalLM:
    if on_gpu:
        
        base_model_loaded = AutoModelForCausalLM.from_pretrained(base_model, token=access_token, quantization_config=bnb_config, device_map={"": 0})
        
    else:
        base_model_loaded = AutoModelForCausalLM.from_pretrained(base_model)

    base_model_loaded.config.use_cache = use_cache
    base_model_loaded.config.pretraining_tp = pretraining_tp

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return base_model_loaded, tokenizer

# for lora and qlora: https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
def prepare_lora_config(r:int=8, lora_alpha:int = 8, lora_dropout:float=.05, bias='none', targets:str='linear', task_type:str='CAUSAL_LM'): # can also take attn
    assert targets in ['linear', 'attn'], "Targets must be 'linear' or 'attn'."
    if targets == 'linear':  # per literature review, best performance is when LoRA and QLoRA are applied to lora linear layers
        target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
    elif targets == 'attn':
        target_modules = ["q_proj", "v_proj"]

    return LoraConfig(r=r, target_modules=target_modules, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias, task_type=task_type)


# for IA3: https://huggingface.co/docs/peft/en/package_reference/ia3
def prepare_ia3_config(r:int=8, targets:str='linear', feedforward_modules=None, task_type:str='CAUSAL_LM'): # can also take attn
    assert targets in ['linear', 'attn'], "Targets must be 'linear' or 'attn'."
    if targets == 'linear':
        target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
    elif targets == 'attn':
        target_modules = ["q_proj", "v_proj"]

    return IA3Config(peft_type="IA3", task_type=task_type, target_modules=target_modules, feedforward_modules=feedforward_modules)


# for AdaLora: https://huggingface.co/docs/peft/en/package_reference/adalora
def prepare_adalora_config(r:int=8, lora_alpha:int = 8, lora_dropout:float=.05, bias='none', targets:str='linear', task_type:str='CAUSAL_LM'): # can also take attn
    assert targets in ['linear', 'attn'], "Targets must be 'linear' or 'attn'."
    if targets == 'linear':  # per literature review, best performance is when LoRA and QLoRA are applied to lora linear layers
        target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
    elif targets == 'attn':
        target_modules = ["q_proj", "v_proj"]

    return AdaLoraConfig(peft_type="ADALORA", task_type=task_type, r=r, target_modules=target_modules, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias)


# https://huggingface.co/docs/peft/en/package_reference/prompt_tuning
# https://huggingface.co/docs/peft/main/en/task_guides/clm-prompt-tuning
def prepare_prompt_tuning_config(task_type:str='CAUSAL_LM', num_virtual_tokens:int = 8, prompt_tuning_init_task:str = None, tokenizer_model:AutoTokenizer=None):

    return PromptTuningConfig(task_type=task_type, prompt_tuning_init="TEXT", num_virtual_tokens=num_virtual_tokens, prompt_tuning_init_text=prompt_tuning_init_task, tokenizer_name_or_path=tokenizer_model)


def prepare_peft_model(base_model:AutoModelForCausalLM, tokenizer:AutoTokenizer, use_cache:bool=False) -> PeftModel: # For LoRA and QLoRA. To run with QLoRA load model in 4bit quantization
    peft_model = prepare_model_for_kbit_training(base_model)
    peft_model.config.pad_token_id = tokenizer.pad_token_id
    peft_model.use_cache = use_cache

    return peft_model


def del_model_of_gpu(model_on_cuda):
    '''
    Deletes model from GPU and clears all the Cache!
    '''
    del model_on_cuda
    gc.collect()
    torch.cuda.empty_cache()


def setup_trainer(model, ds, tokenizer, peft_config, custom_args=None):

    default_args = {
        "output_dir": "./results_qlora",
        "evaluation_strategy": "steps",
        "do_eval": True,
        "optim": "paged_adamw_8bit",
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "log_level": "debug",
        "save_steps": 50,
        "logging_steps": 50,
        "learning_rate": 2e-5,
        "eval_steps": 50,
        "max_steps": 300,
        "warmup_steps": 30,
        "lr_scheduler_type": "linear",
    }

    if custom_args:
        default_args.update(custom_args)

    training_arguments = TrainingArguments(**default_args)


    trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        eval_dataset=ds['dev'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    return trainer


def speculative_decoding(model, assistant_model, inputs, tokenizer):
    outputs = model.generate(**inputs, assistant_model=assistant_model)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def throughput(model, assistant_model, tokenizer, inputs, max_new_tokens=200, temperature=.5):
    start = time.time()
    response = model.generate(**inputs, assistant_model=assistant_model, max_new_tokens=max_new_tokens, temperature=temperature)
    end = time.time()

    latency = end - start
    print(f"Latency: {latency} seconds")

    output_tokens = len(response[0])
    through_put = output_tokens / latency
    print(f"Throughput: {through_put} tokens/second")

    text = tokenizer.decode(response[0])
    print(text)