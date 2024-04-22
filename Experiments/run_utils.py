import os
import re
import json
import torch
import logging 
import pandas as pd


from collections import defaultdict
from datasets import Dataset
import accelerate
import bitsandbytes

from peft import LoraConfig, LoraModel, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from trl import SFTTrainer


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


def load_model(base_model: str, bnb_config:BitsAndBytesConfig=None, on_gpu:bool=False, use_cache:bool=False, pretraining_tp:int=1) -> AutoModelForCausalLM:
    if on_gpu:
        base_model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map={"": 0})
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config)
    
    base_model.config.use_cache = use_cache
    base_model.config.pretraining_tp = pretraining_tp

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return base_model, tokenizer

# for lora and qlora: https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms 
def prepare_lora_config(r:int=8, lora_alpha:int = 8, lora_dropout:float=.05, bias=None, targets:str='linear', task_type:str='CAUSAL_LM'): # can also take attn
    assert targets in ['linear', 'attn'], "Targets must be 'linear' or 'attn'."
    if targets == 'linear':
        target_modules = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']
    elif targets == 'attn':
        target_modules = ["q_proj", "v_proj"]
    
    return LoraConfig(r=r, target_modules=target_modules, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias, task_type=task_type)


def prepare_peft_model(base_model:AutoModelForCausalLM, tokenizer:AutoTokenizer, use_cache=False) -> PeftModel: # For LoRA and QLoRA. To run with QLoRA load model in 4bit quantization
    peft_model = prepare_model_for_kbit_training(base_model)
    peft_model.config.pad_token_id = tokenizer.pad_token_id
    peft_model.use_cache = use_cache


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
        eval_dataset=ds['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    return trainer
