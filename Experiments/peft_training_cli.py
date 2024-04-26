'''
Command Line Tool to allow one to run this on a GPU-As-A-Service Provider such as Lambda Labs or Hyperstack for a single GPU
'''

import argparse
from run_utils import *

def peft_experiments():
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PEFT-based methods with Hugging Face models.")
    parser.add_argument('--base_model', type=str, required=True, choices=['gemma_7b', 'llama2_7b', 'mistral_7b'], help='The base model to use')
    parser.add_argument('--quantization', type=str, required=False, default='determine_optimal', choices=['bnb_4bits', '4bits_nested', '4bits_norm', '4bits_norm_nested', 'determine_optimal'], help='The quantization technique to use. Default is to determine the optimal quantinization technique')
    parser.add_argument('--experiment_type', type=str, required=True, default='all', choices=['all', 'lora', 'qlora', 'ia3', 'adalora', 'prompt_tuning'], help='The type of Parameter Efficient Finetuning (PEFT) Method to use for experimentation. Default will run experiments on all types.')
    parser.add_argument('--dir_path', type=str, required=True, help="Path to the tokenized datasets. Files must be named 'train.json', 'dev.json', 'test.json'")
    parser.add_argument('--on_gpu', action='store_true', help='Flag to run the model on GPU.')
    parser.add_argument('--use_cache', action='store_false', help='Flag to enable caching in the model. Cache is not enabled by default.')
    
    
    args = parser.parse_args()

    # Load the tokenized dataset
    if args.dir_path:
        dataset = load_tokenized_dataset(args.dir_path)
    else:
        raise ValueError("File path for the tokenized dataset is required.")
    

