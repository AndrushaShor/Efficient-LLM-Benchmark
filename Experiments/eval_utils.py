import torch
import evaluate
import datasets
from datasets import Dataset
from trl import SFTTrainer
from transformers import AutoTokenizer
rouge = evaluate.load("rouge")
cosine_similarity = evaluate.load("bertscore")


def compute_accuracy(scores:list):
    num_correct = []
    for score in scores:
        if score == 1:
            num_correct.append(1)
        else:
            num_correct.append(0)
    
    accuracy = 100.0 * (sum(num_correct) / len(num_correct))
    return accuracy


def strip_output_text(output:str, model_name:str):
    model_to_insert_point = {
        'google/gemma-7b': ["Answer:", "Explanation:"],
        'meta-llama/Llama-2-7b-hf': "<s>",
        'mistralai/Mistral-7B-v0.1': "[INST]"
    }
    # Returns the whole input string as well; cut off this part
    out = output[output.find(model_to_insert_point[model_name][0]):output.find(model_to_insert_point[model_name][1])]
    for repl in ['The answer is:', 'Explanation', '\n']:
        out = out.replace(repl, '')
    return out


def predict(trained_model:SFTTrainer, tokenizer:AutoTokenizer, eval_sample:Dataset, model_name:str):
    reqd_cols = ['id', 'questions', 'answers', 'text', 'input_ids', 'prompt_tokenizations', 'original_dataset']
    assert list(eval_sample.features.keys()) == reqd_cols, f"Eval Data needs the following columsn: {reqd_cols}, but instead has { list(eval_sample.features.keys()) }"

    predictions = []
    for inp in eval_sample['prompt_tokenizations']:
        inp = torch.tensor(inp, dtype=int)
        outp = trained_model.generate(inp, max_new_tokens=20, return_dict_in_generate=True, output_scores=True)
        pred = tokenizer.batch_decode(outp['sequences'], skip_special_tokens=True)
        
        predictions.append(strip_output_text(pred[0], model_name))

    return predictions


def evaluate_predictions(eval_sample:Dataset, predictions):
    
    pred_ds = Dataset.from_dict({'predictions': predictions})

    pred_ds = datasets.concatenate_datasets([eval_sample, pred_ds], axis=1)

    original_datasets = set(pred_ds['original_datasets'])
    filt = {}
    for ds in original_datasets:
        filt[ds] = pred_ds(lambda ex: ex['original_datasets'] == ds)
    
    scores = []
    for ds, data in filt.items():
        scores.append(compute_metrics(ds, pred_ds['predictions'], pred_ds['answers']))
    
    accuracy = compute_accuracy(scores)
    return scores, accuracy




def compute_rouge(predictions:list, ground_truth:list):
    scores = rouge.compute(predictions=predictions, references=ground_truth, use_aggregator=False)
    return scores['rougeL'] # longest common subsequence-based ROUGE
    

def compute_similarity(predictions:list, ground_truth:list):
    scores = cosine_similarity.compute(predictions=predictions, references=ground_truth, model_type="distilbert-base-uncased")
    return scores['f1']



def compute_metrics(original_dataset:str, predictions:list, ground_truth:list):

    ds_metric_map = {
        'ai2_science_elementary': 'cosine_similarity',
        'ai2_science_middle': 'cosine_similarity',
        'arc_easy': 'cosine_similarity',
        'arc_hard': 'cosine_similarity',
        'narrativeqa': 'rouge',
        'openbookqa' : 'cosine_similarity',
        'race_string': 'cosine_similarity'}
    
    assert original_dataset in ds_metric_map, f"Please define a metric mapping for dataset {original_dataset}"
    
    metric = ds_metric_map[original_dataset]
    
    if metric == 'rouge':
        scores = compute_rouge(predictions, ground_truth)
    elif metric == 'cosine_similarity':
        scores = compute_similarity(predictions, ground_truth)
    

    return scores



def preprocess_prompt_icl(hf_model: str, ds: Dataset, experiment, k_shot: int=1, 
               max_k_shot_token_length=200, seed=42, sample: int=1000):
    ds = ds.shuffle(seed=seed)
    eval_sample = ds.select(range(sample))

    loaded_tokenizer = AutoTokenizer.from_pretrained(hf_model, device_map={"": 0})
    
    def filter_by_token_length(example):
        tokens = loaded_tokenizer(example['text'], return_tensors="pt", truncation=False)
        return tokens.input_ids.size(1) <= max_k_shot_token_length
    


    print(f'Running prompt injection for: {experiment}')
    prompt_insert = "Answer this question truthfully:"
    
    if experiment == 'zero_shot':
        prompt_insert = "Answer the question truthfully:"
        results = process_samples(eval_sample, hf_model, prompt_insert, loaded_tokenizer)

    elif experiment == 'k_shot':
        filtered_dataset_for_k_shot =  ds.filter(filter_by_token_length) 
        print(f"Number of examples in the dataset: {len(filtered_dataset_for_k_shot)}")
        if len(filtered_dataset_for_k_shot) < k_shot:
            raise ValueError(f"Dataset has less than {k_shot} examples")
        
        prompt_insert = "Answer the question truthfully. Follow these examples:"
        prompt_insert += "\n".join(filtered_dataset_for_k_shot['questions'][:k_shot])
        prompt_insert += "\n"
        prompt_insert += 'Question:'
        
        results = process_samples(eval_sample, hf_model, prompt_insert, loaded_tokenizer)
    print(results['prompt_tokenizations'])
    eval_sample = datasets.concatenate_datasets([eval_sample, results], axis=1)

    return eval_sample

def process_samples(sample_data, model_name, prompt_insert, tokenizer):
    model_to_insert_point = {
        'google/gemma-7b': "user",
        'meta-llama/Llama-2-7b-hf': "<s>",
        'mistralai/Mistral-7B-v0.1': "[INST]"
    }
    
    original_dataset = []
    new_tokenizations = []

    for example in sample_data:
        text = example['questions']
        insertion_point = text.find(model_to_insert_point[model_name]) + len(model_to_insert_point[model_name])
        new_text = text[:insertion_point] + " " + prompt_insert + " " + text[insertion_point:]
        
        inputs = tokenizer(new_text, return_tensors="pt")  
        original_dataset.append(example['id'].split('-')[0])
        new_tokenizations.append(inputs.input_ids)
    processed_samples = {'prompt_tokenizations': new_tokenizations, 'original_dataset': original_dataset}
    out = Dataset.from_dict(processed_samples)
    print(out['prompt_tokenizations'])
    return out