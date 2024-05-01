#!/bin/bash


# Run the peft_training_cli for gemma_2b
python /Experiments/peft_training_cli.py --base_model gemma_2b --quantization determine_optimal --hf_token Your_Huggingface_Token --data_path ../Efficient-LLM-Benchmark/UnifiedQA Data Curation/tokenized/Gemma --experiment_type qlora --on_gpu True --trainer_config ../Efficient-LLM-Benchmark/Experiments/constants/trainer_config.json

# Run the peft_training_cli for gemma_7b
python /Experiments/peft_training_cli.py --base_model gemma_7b --quantization determine_optimal --hf_token Your_Huggingface_Token --data_path ../Efficient-LLM-Benchmark/UnifiedQA Data Curation/tokenized/Gemma --experiment_type qlora --on_gpu True --trainer_config ../Efficient-LLM-Benchmark/Experiments/constants/trainer_config.json

# Run the peft_training_cli for llama2_7b
python /Experiments/peft_training_cli.py --base_model llama2_7b --quantization determine_optimal --hf_token Your_Huggingface_Token --data_path ../Efficient-LLM-Benchmark/UnifiedQA Data Curation/tokenized/Llama --experiment_type qlora --on_gpu True --trainer_config ../Efficient-LLM-Benchmark/Experiments/constants/trainer_config.json

# Run the peft_training_cli for mistral_7b
python /Experiments/peft_training_cli.py --base_model mistral_7b --quantization determine_optimal --hf_token Your_Huggingface_Token --data_path ../Efficient-LLM-Benchmark/UnifiedQA Data Curation/tokenized/Mistral --experiment_type qlora --on_gpu True --trainer_config ../Efficient-LLM-Benchmark/Experiments/constants/trainer_config.json