'''
Command Line Tool to allow one to run this on a GPU-As-A-Service Provider such as Lambda Labs or Hyperstack for a single GPU
'''

import argparse
from run_utils import *
from quantization import * 

def main(hf_token:str, base_model:str, quantization_type:str, dir_path:str, experiment_type:str, on_gpu:bool, use_cache:bool):
    quant_types = [CONFIG_4BITS, CONFIG_4BITS_NESTED, CONFIG_4BITS_NORM, CONFIG_4BITS_NORM_NESTED] # all possible configs

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

    
    # Load model based on specifications 
    if experiment_type == 'qlora':
        
        determine_optimal = False
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
        
        if determine_optimal:
            for quant_config in quant_types:
                loaded_base_model = load_model(hf_model, quant_config, hf_token, on_gpu=on_gpu, use_cache=use_cache)

        else:
            loaded_base_model = load_model(hf_model, bnb_config, hf_token, on_gpu=on_gpu, use_cache=use_cache)
    elif experiment_type == 'lora':
        loaded_base_model = load_model(hf_model, bnb_config, hf_token, on_gpu=on_gpu, use_cache=use_cache)
    elif experiment_type == 'ia3':
        pass
    elif experiment_type == 'adalora':
        pass
    elif experiment_type == 'prompt_tuning':
        pass
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Run PEFT-based methods with Hugging Face models.")
    parser.add_argument('--hf_token', type=str, required=True, help="Huggingface token in order to access Gemma-2b, Gemma-7b, Mistral-7b, and Llama2-7b")
    parser.add_argument('--base_model', type=str, required=True, choices=['gemma_2b', 'gemma_7b', 'llama2_7b', 'mistral_7b'], help='The base model to use')
    parser.add_argument('--quantization', type=str, required=True, default='base', choices=['4bits', '4bits_nested', '4bits_norm', '4bits_norm_nested', 'determine_optimal', 'base'], help='The quantization technique to use. Default is to load unquantized base model')
    parser.add_argument('--dir_path', type=str, required=True, help="Path to the tokenized datasets. Files must be named 'train.json', 'dev.json', 'test.json'")
    parser.add_argument('--experiment_type', type=str, required=True, default='qlora', choices=['lora', 'qlora', 'ia3', 'adalora', 'prompt_tuning'], help='The type of Parameter Efficient Finetuning (PEFT) Method to use for experimentation. Default will run experiments on all types.')
    parser.add_argument('--on_gpu', action='store_true', help='Flag to run the model on GPU.')
    parser.add_argument('--use_cache', action='store_false', help='Flag to enable caching in the model. Cache is not enabled by default.')
    
    
    args = parser.parse_args()

    # only using bnb config for qlora
    if args.experiment_type != 'qlora':
        print(f'Not loading quantization for {args.experiment_type}')
        args.quantization = 'base' 
    main(base_model=args.base_model, quantization_type=args.quantization, dir_path=args.dir_path,
          experiment_type=args.experiment_type, on_gpu=args.on_gpu, use_cache=args.use_cache)
    
    


    

    

    

