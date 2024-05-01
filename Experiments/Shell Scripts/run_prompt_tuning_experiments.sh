#!/bin/bash

# We could not run this experiment due to limitations on our A100 GPU
# Run the peft_training_cli for gemma_2b
python /home/ubuntu/efficient-llm-finetuning/Efficient-LLM-Benchmark/Experiments/peft_training_cli.py --base_model gemma_2b --quantization 4bits_norm_nested --hf_token Your_Huggingface_Token --data_path /home/ubuntu/efficient-llm-finetuning/Efficient-LLM-Benchmark/"UnifiedQA Data Curation"/tokenized/Gemma --experiment_type prompt_tuning --on_gpu
nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

# Run the peft_training_cli for gemma_7b
python /home/ubuntu/efficient-llm-finetuning/Efficient-LLM-Benchmark/Experiments/peft_training_cli.py --base_model gemma_7b --quantization 4bits_norm_nested --hf_token Your_Huggingface_Token --data_path /home/ubuntu/efficient-llm-finetuning/Efficient-LLM-Benchmark/"UnifiedQA Data Curation"/tokenized/Gemma --experiment_type prompt_tuning --on_gpu

nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

# Run the peft_training_cli for llama2_7b
python /home/ubuntu/efficient-llm-finetuning/Efficient-LLM-Benchmark/Experiments/peft_training_cli.py --base_model llama2_7b --quantization 4bits_norm_nested --hf_token Your_Huggingface_Token --data_path /home/ubuntu/efficient-llm-finetuning/Efficient-LLM-Benchmark/"UnifiedQA Data Curation/tokenized"/Llama --experiment_type prompt_tuning --on_gpu

nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

# Run the peft_training_cli for mistral_7b
python /home/ubuntu/efficient-llm-finetuning/Efficient-LLM-Benchmark/Experiments/peft_training_cli.py --base_model mistral_7b --quantization 4bits_norm_nested --hf_token Your_Huggingface_Token --data_path /home/ubuntu/efficient-llm-finetuning/Efficient-LLM-Benchmark/"UnifiedQA Data Curation"/tokenized/Mistral --experiment_type prompt_tuning --on_gpu
nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9