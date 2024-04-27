'''
Command Line Tool to allow one to run this on a GPU-As-A-Service Provider such as Lambda Labs or Hyperstack for a single GPU
'''

import argparse
import json
from gcp_storage_client import storage_client
from run_utils import *
from quantization import * 


def setup(hf_token:str, base_model:str, quantization_type:str, dir_path:str, experiment_type:str, on_gpu:bool, use_cache:bool, trainer_config_path:str, service_account_path:str, project_id:str, storage_bucket:str):
    quant_types = [CONFIG_4BITS, CONFIG_4BITS_NESTED, CONFIG_4BITS_NORM, CONFIG_4BITS_NORM_NESTED] # all possible bitsandbytes configs
    determine_optimal = False # used for qlora
    
    # Setup storage
    gcp_storage_client = storage_client(service_account_path=service_account_path, project_id=project_id)

    # Get train, dev, and test datasets from dir_path
    train, dev, test = load_datasets_from_directory(dir_path)

    # figure out hf path to model based on base model
    if base_model == 'gemma_2b':
        hf_model = 'google/gemma-2b' # Gemma-2b
    elif base_model == 'gemma_7b': 
        hf_model = 'google/gemma-7b' # Gemma-7b
    elif base_model == 'llama2_7b':
        hf_model = 'meta-llama/Llama-2-7b-hf' # Llama2-7b
    elif base_model == 'mistral_7b':
        hf_model = 'mistralai/Mistral-7B-v0.1' # Mistral-7b 


    # figure out what quantization type to use based on quantization_type
    if quantization_type == 'base':
        bnb_config = None
    if quantization_type == '4bits':
        bnb_config = quant_types[0]
    elif quantization_type == '4bits_nested':
        bnb_config = quant_types[1]
    elif quantization_type == '4bits_norm':
        bnb_config = quant_types[2]
    elif quantization_type == '4bits_norm_nested':
        bnb_config = quant_types[3]
    else:
        determine_optimal = True

    # figure out config to use based on experiment type
    if experiment_type == 'lora' or experiment_type == 'qlora':
        peftConfig = prepare_lora_config(r=8, lora_alpha=16, lora_dropout=0.1, bias='none', targets='linear') 
        peftConfig_attn = prepare_lora_config(r=8, lora_alpha=16, lora_dropout=0.1, bias='none', targets='attn')
    elif experiment_type == 'ia3': 
        peftConfig = prepare_ia3_config(targets='linear') 
        peftConfig_attn = prepare_ia3_config(targets='attn')
    elif experiment_type == 'adalora':
        peftConfig = prepare_adalora_config(r=8, lora_alpha=16, lora_dropout=0.1, bias='none', targets='linear') 
        peftConfig_attn = prepare_adalora_config(r=8, lora_alpha=16, lora_dropout=0.1, bias='none', targets='attn')
    elif experiment_type == 'prompt_tuning':
        peftConfig = prepare_prompt_tuning_config(prompt_tuning_init_task="Let's think step by step.", num_virtual_tokens=10)
        peftConfig_attn = None


    # Read trainer config if it exists
    trainer_config = None
    if trainer_config_path is not None:
        with open(trainer_config_path) as fp:
            trainer_config = json.load(fp)

    if not determine_optimal: # train_peft
        loaded_base_model, loaded_tokenizer = load_model(base_model=hf_model, bnb_config=bnb_config, access_token=hf_token, on_gpu=on_gpu, use_cache=use_cache)
        ds = {'train': train, 'dev': dev, 'test': test}
        train_peft(loaded_base_model=loaded_base_model, loaded_tokenizer=loaded_tokenizer, peftConfig=peftConfig, trainer_config=trainer_config, ds=ds)
        train_peft(loaded_base_model=loaded_base_model, loaded_tokenizer=loaded_tokenizer, peftConfig=peftConfig_attn, trainer_config=trainer_config, ds=ds)


    else: # run quantization experiments
        for quant_config in quant_types: # qlora
            loaded_base_model, loaded_tokenizer = load_model(base_model=hf_model, bnb_config=quant_config, access_token=hf_token, on_gpu=on_gpu, use_cache=use_cache)
            ds = {'train': train, 'dev': dev, 'test': test}
            train_peft(loaded_base_model=loaded_base_model, loaded_tokenizer=loaded_tokenizer, peftConfig=peftConfig, trainer_config=trainer_config, ds=ds)
            train_peft(loaded_base_model=loaded_base_model, loaded_tokenizer=loaded_tokenizer, peftConfig=peftConfig_attn, trainer_config=trainer_config, ds=ds)



def train_peft(loaded_base_model, loaded_tokenizer, peftConfig, trainer_config, ds): 
    assert peftConfig != None, 'Cannot train with PEFT Methods without PEFT Config!'
    

    peft_model = prepare_peft_model(loaded_base_model, loaded_tokenizer)
    print(peft_model.print_trainable_parameters())
    
    trainer = setup_trainer(model=peft_model, ds=ds, tokenizer=loaded_tokenizer,  peftConfig=peftConfig, custom_args=trainer_config)
    trainer.train()


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
    parser.add_argument('--service_account_path', type=str, required=True, help='Service Account location to be able to connect to GCP Storage Buckets')
    parser.add_arguments('--project_id', type=str, required=True, help='gcp project id to use')
    parser.add_argument('--storage_bucket', type=str, required=True, help='Location to store the trained model and the checkpoints') 

    

    
    args = parser.parse_args()

    if torch.cuda.is_available():
        # only using bnb config for qlora
        if args.experiment_type != 'qlora':
            print(f'Not loading quantization for {args.experiment_type}')
            args.quantization = 'base' 
        setup(base_model=args.base_model, quantization_type=args.quantization, dir_path=args.data_path,
            experiment_type=args.experiment_type, on_gpu=args.on_gpu, use_cache=args.use_cache)
    else:
        print('Please make sure you have a GPU Available!')
    


    

    

    

