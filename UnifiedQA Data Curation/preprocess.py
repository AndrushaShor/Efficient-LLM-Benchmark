import os
import json
import re
import string
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def make_unified_qa_dataset(unified_datasets: list, data_path: str, data_type: str, enable_load:bool=False, file_path:str=None) -> dict:
    assert data_type in ["train", "dev", "test"], "data_type must be 'train', 'dev', or 'test'"
    assert not enable_load or file_path is not None, "preprocessed_path cannot be None if enable_load is True"
    
    unified_dataset = {}
    for dataset in unified_datasets:
        
        curr_data_path = os.path.join(data_path, dataset, f"{data_type}.tsv")
        assert os.path.isfile(curr_data_path), f"Data file does not exist: {curr_data_path}"
        unified_dataset[dataset] = {"id": [], "question": [], "answer": []}
        
        with open(curr_data_path, "r", encoding="utf-8") as f:
            cnt = 0
            for line in f:
                if line.strip():  
                    question, answer = line.strip().split("\t")
                    unified_dataset[dataset]["id"].append(f"{dataset}-{data_type}-{cnt}")
                    unified_dataset[dataset]["question"].append(question)
                    unified_dataset[dataset]['answer'].append(answer)
                    cnt += 1
    if enable_load:
        id, questions, answers = [], [], []
        for dataset in unified_dataset.keys():
            id += unified_dataset[dataset]['id']
            questions += unified_dataset[dataset]['question']
            answers += unified_dataset[dataset]['answer']
        
        os.makedirs(file_path, exist_ok=True)
        fp = file_path + "\\" + data_type + ".json"
        with open(fp, "w") as f:
            json.dump([id, questions, answers], f)
    return unified_dataset


def preprocess_unified_qa_dataset(datasets: dict, append_instruction_gemma: bool=False, append_instruction_llama: bool=False, 
                                  append_instruction_mistral: bool=False, append_bos: bool=False, append_s :bool=False, enable_load:bool=False, 
                                  file_path:str=None, file_name:str=None) -> dict:
    
    assert not enable_load or file_path is not None or file_name is not None, "file_path cannot be None if enable_load is True"

    assert sum([append_instruction_gemma, append_instruction_llama, append_instruction_mistral]) == 1, \
    "Exactly one 'append_instruction...' parameter must be true"
    
    assert sum([append_bos, append_s]) <= 1, \
    "At most one of 'append_bos' or 'append_s' can be true"

    preprocessed_unified_dataset = {}

    for dataset in datasets.keys():
        preprocessed_unified_dataset[dataset] = {"id": [], "question": [], "answer": []}
        preprocessed_unified_dataset[dataset]['id'] = [id for id in datasets[dataset]['id']]
        
        preprocessed_unified_dataset[dataset]['question'] = [question.lower().strip() for question in datasets[dataset]['question']]
        preprocessed_unified_dataset[dataset]['answer'] = [answer.lower().strip() for answer in datasets[dataset]['answer']]
        if append_instruction_gemma: # For Gemma Models: https://huggingface.co/google/gemma-7b/discussions/62
            preprocessed_unified_dataset[dataset]['question'] = ["<start_of_turn>user\n" + question + "<end_of_turn>\n\n" for question in preprocessed_unified_dataset[dataset]['question']]
            preprocessed_unified_dataset[dataset]['answer'] = ["<start_of_turn>model\n" + answer + "<end_of_turn>" for answer in preprocessed_unified_dataset[dataset]['answer']]
        if append_instruction_llama: # For Llama 2: https://huggingface.co/docs/optimum-neuron/en/tutorials/fine_tune_llama_7b or https://github.com/mallorbc/llama_dataset_formats/blob/26b29649dca39552e2ecb9d7041468488b9b0f32/README.md
            preprocessed_unified_dataset[dataset]['answer'] = ["Output:\n" + answer for answer in preprocessed_unified_dataset[dataset]['answer']]
        if append_instruction_mistral: # For Mistral 7b: https://www.promptingguide.ai/models/mistral-7b
            preprocessed_unified_dataset[dataset]['question'] = ["[INST] " + question + " [/INST]" for question in preprocessed_unified_dataset[dataset]['question']] 
            preprocessed_unified_dataset[dataset]['answer'] = [answer for answer in preprocessed_unified_dataset[dataset]['answer']]


        if append_bos:
            preprocessed_unified_dataset[dataset]['question'] = ["<bos>"+question for question in preprocessed_unified_dataset[dataset]['question']]
        if append_s:
            preprocessed_unified_dataset[dataset]['question'] = ["<s>"+question + '\n\n' for question in preprocessed_unified_dataset[dataset]['question']]
        
        preprocessed_unified_dataset[dataset]['text'] = [q + " " + a for q, a in zip(preprocessed_unified_dataset[dataset]["question"], preprocessed_unified_dataset[dataset]["answer"])]



    if enable_load:
        id, questions, answers, text = [], [], [], []
        for dataset in preprocessed_unified_dataset.keys():
            id += preprocessed_unified_dataset[dataset]['id']
            questions += preprocessed_unified_dataset[dataset]['question']
            answers += preprocessed_unified_dataset[dataset]['answer']
            text += preprocessed_unified_dataset[dataset]['text']


        os.makedirs(file_path, exist_ok=True)
        fp = file_path + "\\" + file_name + '.json'
        print(fp)
        with open(fp, "w") as f:
            json.dump([id, questions, answers], f)

    return preprocessed_unified_dataset


def tokenize_dataset(preprocessed_unified_dataset: dict, model_name: str, pad_token: bool, pad_side:str='right', enable_load:bool=True, 
                 file_path:str=None, file_name:str=None) -> dict:
    assert not enable_load or file_path is not None or file_name is not None, "file_path cannot be None if enable_load is True"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = pad_side
    for dataset in preprocessed_unified_dataset.keys():
        preprocessed_unified_dataset[dataset]['input_ids'] = [tokenizer(item, return_tensors='pt').input_ids.tolist() for item in preprocessed_unified_dataset[dataset]['text']]

    if enable_load:
        id, questions, answers, text, input_id = [], [], [], [], []
        for dataset in preprocessed_unified_dataset.keys():
            id += preprocessed_unified_dataset[dataset]['id']
            questions += preprocessed_unified_dataset[dataset]['question']
            answers += preprocessed_unified_dataset[dataset]['answer']
            text += preprocessed_unified_dataset[dataset]['text']
            input_id += preprocessed_unified_dataset[dataset]['input_ids']
        
        
        os.makedirs(file_path, exist_ok=True)
        fp = file_path + "\\" + file_name + '.json'
        with open(fp, "w") as f:
            json.dump([id, questions, answers, text, input_id], f)

    return preprocessed_unified_dataset




