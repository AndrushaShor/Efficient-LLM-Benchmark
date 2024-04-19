import os
import json
import torch
import logging 
import pandas as pd

from collections import defaultdict
from datasets import Dataset
import accelerate
import bitsandbytes

from peft import LoraConfig, LoraModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from trl import SFTTrainer



def load_tokenized_dataset(self, name:str, file_path:str):
    data_dict = {}
    with open(file_path, 'r') as fp:
        id, questions, answers, text, input_id = json.load(fp)
        
        data_dict['id'] = id
        data_dict['questions'] = questions
        data_dict['answers'] = answers
        data_dict['text'] = text
        data_dict['input_ids'] = input_id
        
        self.dataset[name] = Dataset.from_dict(data_dict)
    
    return Dataset.from_dict(data_dict)



