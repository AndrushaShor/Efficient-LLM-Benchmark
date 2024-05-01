'''
Command Line Tool to allow one to run this on a GPU-As-A-Service Provider such as Lambda Labs or Hyperstack for a single GPU
'''

import argparse
import gc
import os
import json
import peft
import torch
import transformers
from gcp_storage_client import storage_client
from run_utils import *
from quantization import * 


def setup(hf_token:str, base_model:str, quantization_type:str, dir_path:str, experiment_type:str, on_gpu:bool, use_cache:bool, trainer_config_path:str=None, service_account_path:str=None, project_id:str=None, storage_bucket:str=None):
    quant_types = [CONFIG_4BITS, CONFIG_4BITS_NESTED, CONFIG_4BITS_NORM, CONFIG_4BITS_NORM_NESTED] # all possible bitsandbytes configs
    determine_optimal = False # used for qlora
    
    # Setup storage
    # gcp_storage_client = storage_client(service_account_path=service_account_path, project_id=project_id)

    # Get train, dev, and test datasets from dir_path
    train, dev, test = load_datasets_from_directory(dir_path)
    # Remove bad input_ids
    ds = {'train': train, 'dev': dev, 'test': test}
    print(ds['train'])
    ds['train'] = ds['train'].remove_columns(['input_ids'])
    ds['dev'] = ds['dev'].remove_columns(['input_ids'])
    ds['train'] = ds['train'].shuffle(seed=42)
    ds['dev'] = ds['dev'].shuffle(seed=42)
    
    # figure out hf path to model based on base model
    if base_model == 'gemma_2b':
        hf_model = 'google/gemma-2b' # Gemma-2b
    elif base_model == 'gemma_7b': 
        hf_model = 'google/gemma-7b' # Gemma-7b
    elif base_model == 'llama2_7b':
        hf_model = 'meta-llama/Llama-2-7b-hf' # Llama2-7b
    elif base_model == 'mistral_7b':
        hf_model = 'mistralai/Mistral-7B-v0.1' # Mistral-7b 
    print(f"Base model: {hf_model}")

    # figure out what quantization type to use based on quantization_type
    if quantization_type == 'base':
        bnb_config = None
    elif quantization_type == '4bits':
        bnb_config = quant_types[0]
    elif quantization_type == '4bits_nested':
        bnb_config = quant_types[1]
    elif quantization_type == '4bits_norm':
        bnb_config = quant_types[2]
    elif quantization_type == '4bits_norm_nested':
        bnb_config = quant_types[3]
    else:
        determine_optimal = True

    print(f"Quantization Type: {quantization_type}")
    print(f"BNB Config: {bnb_config}")
    print(f"")
    # figure out config to use based on experiment type
    if experiment_type == 'lora' or experiment_type == 'qlora':
        print("Preparing Lora or QLora")
        peftConfig = prepare_lora_config(r=8, lora_alpha=16, lora_dropout=0.1, bias='none', targets='linear') 
        peftConfig_attn = prepare_lora_config(r=8, lora_alpha=16, lora_dropout=0.1, bias='none', targets='attn')
    elif experiment_type == 'ia3': 
        print("Preparing ia3")
        peftConfig = prepare_ia3_config(targets='linear') 
        peftConfig_attn = prepare_ia3_config(targets='attn')
    elif experiment_type == 'adalora':
        print("Preparing adalora")
        peftConfig = prepare_adalora_config(r=8, lora_alpha=16, lora_dropout=0.1, bias='none', targets='linear') 
        peftConfig_attn = prepare_adalora_config(r=8, lora_alpha=16, lora_dropout=0.1, bias='none', targets='attn')
    elif experiment_type == 'prompt_tuning':
        print("Preparing Prompt Tuning")
        peftConfig = prepare_prompt_tuning_config(prompt_tuning_init_task="Let's think step by step.", num_virtual_tokens=10, tokenizer_model = hf_model)
        peftConfig_attn = None


    # Read trainer config if it exists
    trainer_config = None
    if trainer_config_path is not None:
        with open(trainer_config_path) as fp:
            trainer_config = json.load(fp)

    
    if not determine_optimal: # train_peft
        print(f"Running: {base_model}_{experiment_type}_{quantization_type}")
        
        model_name_linear = f'{base_model}_{experiment_type}_{quantization_type}_linear'
        # model_name_attn = f'{base_model}_{experiment_type}_{quantization_type}_attn'
        loaded_base_model, loaded_tokenizer = load_model(base_model=hf_model, bnb_config=bnb_config, access_token=hf_token, on_gpu=on_gpu, use_cache=use_cache)
        
        
        train_peft(loaded_base_model=loaded_base_model, loaded_tokenizer=loaded_tokenizer, peftConfig=peftConfig, trainer_config=trainer_config, ds=ds, model_name=model_name_linear)
        # train_peft(loaded_base_model=loaded_base_model, loaded_tokenizer=loaded_tokenizer, peftConfig=peftConfig_attn, trainer_config=trainer_config, ds=ds, model_name=model_name_attn)


    else: # run quantization experiments

        print("Running determine optimal quant config")
        for i, quant_config in enumerate(quant_types): # qlora
            
            quant_names = ['4bits', '4bits_nested', '4bits_norm', '4bits_norm_nested']
            print(f"Running: {base_model}_{experiment_type}_{quant_names[i]}")
            model_name = f'{base_model}_{experiment_type}_{quant_names[i]}'
            loaded_base_model, loaded_tokenizer = load_model(base_model=hf_model, bnb_config=quant_config, access_token=hf_token, on_gpu=on_gpu, use_cache=use_cache)
            #ds = {'train': train, 'dev': dev, 'test': test}
            train_peft(loaded_base_model=loaded_base_model, loaded_tokenizer=loaded_tokenizer, peftConfig=peftConfig, trainer_config=trainer_config, ds=ds, model_name=model_name)
            train_peft(loaded_base_model=loaded_base_model, loaded_tokenizer=loaded_tokenizer, peftConfig=peftConfig_attn, trainer_config=trainer_config, ds=ds, model_name=model_name)



def train_peft(loaded_base_model, loaded_tokenizer, peftConfig, trainer_config, ds, model_name, seed=42, eval_samples=200): 
    assert peftConfig != None, 'Cannot train with PEFT Methods without PEFT Config!'

    gc.collect()
    torch.cuda.empty_cache() # Remove any cache on the GPU 


    
    dir_fp = model_name + '_outputs/' # for unique directory for checkpoints
    os.makedirs(dir_fp, exist_ok=True)
    
    fp = dir_fp + "/" + model_name + '_final' # model_name
    tokenizer_name = fp + '_tokenizer' # '
    metrics_log = fp + '_metrics.json'
    
    util_log = fp + '_trainable_params.txt'
    peft_model = prepare_peft_model(loaded_base_model, loaded_tokenizer)

    
    trainer = setup_trainer(model=peft_model, ds=ds, tokenizer=loaded_tokenizer, output_dir=dir_fp, peft_config=peftConfig, custom_args=trainer_config)

    
    trainable_params = print_trainable_parameters(peft_model)
    with open(util_log, 'w') as f:
        f.write(trainable_params)

    # conducting training
    trainer.train()
    trainer.save_model(fp)
    
    
    trainer.save_model(tokenizer_name)

    
    with open(metrics_log, 'w') as f:
        json.dump(trainer.state.log_history, f)  
        
def upload_to_gcp(service_account_path:str, project_id:str, storage_bucket:str):
    client = storage_client(service_account_path=service_account_path, project_id=project_id)
    client.upload_directory(storage_bucket, 'results')
    # # Upload files in outputs directory to GCP
    # for file_name in os.listdir('outputs'):
    #     file_path = os.path.join('outputs', file_name)
    #     storage_client.upload_blob(bucket_name=storage_bucket, file_path=file_path, obj_name=file_name)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run PEFT-based methods with Hugging Face models. This CLI requires a GPU to run!")
    # HF Params
    parser.add_argument('--base_model', type=str, required=True, choices=['gemma_2b', 'gemma_7b', 'llama2_7b', 'mistral_7b'], help='The base model to use')
    parser.add_argument('--quantization', type=str, required=True, default='base', choices=['4bits', '4bits_nested', '4bits_norm', '4bits_norm_nested', 'determine_optimal', 'base'], help='The quantization technique to use. Default is to load unquantized base model')
    parser.add_argument('--hf_token', type=str, required=True, help="Huggingface token in order to access Gemma-2b, Gemma-7b, Mistral-7b, and Llama2-7b")
    
    # Experiment Params
    parser.add_argument('--data_path', type=str, required=True, help="Path to the tokenized datasets. Files must be named 'train.json', 'dev.json', 'test.json'")
    parser.add_argument('--experiment_type', type=str, required=True, default='qlora', choices=['lora', 'qlora', 'ia3', 'adalora', 'prompt_tuning'], help='The type of Parameter Efficient Finetuning (PEFT) Method to use for experimentation. Default will run experiments on all types.')
    parser.add_argument('--on_gpu', action='store_true', help='Flag to run the model on GPU.')
    parser.add_argument('--use_cache', action='store_false', help='Flag to enable caching in the model. Cache is not enabled by default.')
    parser.add_argument('--trainer_config', type=str, default=None, required=False, help='path to custom trainer config')
    
    # GCP Params
    parser.add_argument('--service_account_path', type=str, required=False, default=None, help='Service Account location to be able to connect to GCP Storage Buckets')
    parser.add_argument('--project_id', type=str, required=False, default=None, help='gcp project id to use')
    parser.add_argument('--storage_bucket', type=str, required=False, help='Location to store the trained model and the checkpoints') 

    
    args = parser.parse_args()

    # Check to see if user wants to upload results to GCP
    if args.service_account_path is not None:
        assert args.project_id is not None, "Please provide a project_id if you are using a service account"
        assert args.storage_bucket is not None, "Please provide a storage bucket if you are using a service account"

        upload_to_gcp(service_account_path=args.service_account_path, project_id=args.project_id, storage_bucket=args.storage_bucket)

    # Otherwise user wants to run experiments
    elif torch.cuda.is_available():
        # only using bnb config for qlora
        # if args.experiment_type != 'qlora':
        #     print(f'Not loading quantization for {args.experiment_type}')
        #     args.quantization = 'base' 
        setup(hf_token=args.hf_token, base_model=args.base_model, quantization_type=args.quantization, dir_path=args.data_path,
            experiment_type=args.experiment_type, on_gpu=args.on_gpu, use_cache=args.use_cache)
    
    # otherwise user does not have a GPU
    else:
        print('Please make sure you have a GPU and CUDA installed!')
    


    

    

    

