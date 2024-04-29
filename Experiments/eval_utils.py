import os
import time
import json
import torch
import evaluate
import datasets
from datasets import Dataset
from trl import SFTTrainer
from transformers import AutoTokenizer
rouge = evaluate.load("rouge")
cosine_similarity = evaluate.load("bertscore")
perplexity = evaluate.load("perplexity", module_type="metric")



# Prompt ICL functions
def preprocess_prompt_icl(hf_model: str, loaded_tokenizer:AutoTokenizer, ds: Dataset, experiment, k_shot: int=1,
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


# Prediction functions

def predict(trained_model:SFTTrainer, tokenizer:AutoTokenizer, eval_sample:Dataset, model_name:str, prompted:bool=False):
    if prompted==True:
        assert 'prompt_tokenizations' in list(eval_sample.features.keys()), f"Eval Data needs the following column: 'prompt_tokenizations', but instead has { list(eval_sample.features.keys()) }"
        token_col = 'prompt_tokenizations'
    else:
        assert 'input_ids' in list(eval_sample.features.keys()), f"Eval Data needs the following column: 'input_ids', but instead has { list(eval_sample.features.keys()) }"
        token_col = 'input_ids'
    eval_sample = eval_sample.with_format("torch")
    predictions = []
    latencies = []
    for inp in eval_sample[token_col]:
        # inp = torch.tensor(inp, dtype=int)
        start = time.time()
        outp = trained_model.generate(inp, max_new_tokens=20, return_dict_in_generate=True, output_scores=True)
        end = time.time()
        pred = tokenizer.batch_decode(outp['sequences'], skip_special_tokens=True)

        predictions.append(pred[0])
        latencies.append(end - start)

    return predictions, latencies


def strip_output_text(output:str, model_name:str):
  processed = output
  if model_name == 'google/gemma-7b':
    if 'Answer:' in processed:
      processed = processed[processed.find("Answer:")+7:]
    if 'model' in processed:
      processed = processed[processed.find('model')+5:]
    # Returns the whole input string as well; cut off this part
    for repl in ['\n',  'Step 1/', 'Step 2/',]:
        processed = processed.replace(repl, '')
    for strp in ['The answer is:']:
        processed = processed.strip(strp)

  return processed



def strip_answers(answer_text:str, model_name:str):
  out = answer_text
  if model_name == 'google/gemma-7b':
    for strp in ['<start_of_turn>model\n', '<end_of_turn>']:
      out = out.replace(strp, '')
  elif model_name == 'meta-llama/Llama-2-7b-hf':
    out = []
  elif model_name == 'mistralai/Mistral-7B-v0.1':
    out = []
  return out


def prediction_wrapper(trained_model:SFTTrainer, tokenizer:AutoTokenizer, ds:Dataset, model_name:str, add_prompt:bool=False, sample:int=1000, seed:int=42, save_path:str=''):
    def add_dataset_name_col(ds):
        original_dataset = []
        for example in ds:
            original_dataset.append(example['id'].split('-')[0])
        eval_sample = datasets.concatenate_datasets([ds, Dataset.from_dict({'original_dataset': original_dataset})], axis=1)
        return eval_sample

    if add_prompt == True and sample > 0:
         eval_sample = preprocess_prompt_icl(model_name, tokenizer, ds, experiment='zero_shot', sample=sample, seed=seed)
    elif add_prompt == False and sample > 0:
        ds = ds.shuffle(seed=seed)
        sample_data = ds.select(range(sample))
        eval_sample = add_dataset_name_col(sample_data)
    elif add_prompt == True and sample == 0:
         eval_sample = preprocess_prompt_icl(model_name, tokenizer, ds, experiment='zero_shot', sample=ds.shape[0], seed=seed)
    else:
        eval_sample = add_dataset_name_col(ds)
    print("eval_sample generated")
    predictions, latencies = predict(trained_model, tokenizer, eval_sample, model_name, prompted=add_prompt)
    print("predictions generated")
    # predictions = [predictions[i][len(eval_sample['questions'][i]):] for i in range(len(eval_sample['questions']))]
    predictions = [strip_output_text(s, model_name) for s in predictions]

    answers_stripped = [strip_answers(s, model_name) for s in eval_sample['answers']]


    pred_ds = Dataset.from_dict({
        'predictions': [p.lower() for p in predictions],
        'ground_truth':answers_stripped,
        'original_dataset':eval_sample['original_dataset'],
        'latencies': latencies})

    if len(save_path) > 0:

        dir = save_path.split('/')[:-1]
        print(os.path.join(*dir))
        os.makedirs(f"/{os.path.join(*dir)}", exist_ok=True)
        print(save_path)
        with open(save_path, "w") as f:
            json.dump([pred_ds['predictions'], pred_ds['ground_truth'], pred_ds['original_dataset'], pred_ds['latencies']], f)

    return pred_ds


# Metrics
def compute_accuracy(scores:list):
    num_correct = 0
    for score in scores:
        if score == 1:
            num_correct += 1

    accuracy = 100.0 * ((num_correct) / len(scores))
    return accuracy


def compute_ppl(predictions, trained_model):
    # This is currently wrong; needs a string for model_id that points to a TRAINED model (which we don't have yet)
    ppl = perplexity.compute(predictions=predictions, model_id=trained_model)['mean_perplexity']
    return ppl

def throughput(latencies:list, predictions:list):
    print('computing throughput')
    through_put = []
    for l, p in zip(latencies, predictions):
        output_tokens = len(p)
        through_put.append(output_tokens / l)

    avg_through_put = sum(through_put) / len(through_put)
    return avg_through_put

def compute_rouge(predictions:list, ground_truth:list):
    print('computing similarity for summarization')
    scores = rouge.compute(predictions=predictions, references=ground_truth, use_aggregator=False)
    return scores['rougeL'] # longest common subsequence-based ROUGE


def jaccard(str1, str2):
    if str1 == str2:
        return 1.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        return 0.0

            
def compute_similarity(predictions:list, ground_truth:list):
    print('computing similarity for multiple choice')
    # scores = cosine_similarity.compute(predictions=predictions, references=ground_truth, model_type="distilbert-base-uncased")
    # scores = scores['f1']
    scores = []
    for p, l in zip(predictions, ground_truth):
      scores.append(jaccard(p,l))
    return scores


def compute_scores(original_dataset:str, predictions:list, ground_truth:list):

    ds_metric_map = {
        'ai2_science_elementary': 'mc',
        'ai2_science_middle': 'mc',
        'arc_easy': 'mc',
        'arc_hard': 'mc',
        'narrativeqa': 'rouge',
        'openbookqa' : 'mc',
        'race_string': 'mc'}

    assert original_dataset in ds_metric_map, f"Please define a metric mapping for dataset {original_dataset}"

    metric = ds_metric_map[original_dataset]

    if metric == 'rouge':
        scores = compute_rouge(predictions, ground_truth)
    elif metric == 'mc':
        scores = compute_similarity(predictions, ground_truth)

    return scores

# Evaluation functions
def evaluate_predictions(pred_ds:Dataset, model_name:str):

    assert list(pred_ds.features.keys()) == ['predictions', 'ground_truth', 'original_dataset', 'latencies'], f"Prediction dataset must have ['predictions', 'ground_truth', 'original_dataset'] in columns, currently has {list(pred_ds.features.keys()) }."


    original_datasets = set(pred_ds['original_dataset'])
    filt = {}
    for ds in original_datasets:
        filt[ds] = pred_ds.filter(lambda ex: ex['original_dataset'] == ds)

    scores = []
    for ds, data in filt.items():
        scores.extend(compute_scores(ds, pred_ds['predictions'], pred_ds['ground_truth']))

    accuracy = compute_accuracy(scores)

    print("computing perplexity")

    tp = throughput(pred_ds['latencies'], pred_ds['predictions'])

    return scores, accuracy, tp

def load_saved_predictions(file_path:str):
    data_dict = {}
    with open(file_path, 'r') as fp:
        predictions, ground_truth, original_dataset, latencies = json.load(fp)

        data_dict['predictions'] = predictions
        data_dict['ground_truth'] = ground_truth
        data_dict['original_dataset'] = original_dataset
        data_dict['latencies'] = latencies

    return Dataset.from_dict(data_dict)


def predict_and_evaluate(trained_model:SFTTrainer, tokenizer:AutoTokenizer, ds:Dataset, model_name:str, add_prompt:bool=False, sample:int=1000, seed:int=42, return_predictions:bool=False):
      print("calculating predictions")
      pred_ds = prediction_wrapper(trained_model, tokenizer, ds, model_name, add_prompt, sample, seed)
      print("calculating metrics")
      metrics = evaluate_predictions(pred_ds, model_name)

      if return_predictions:
        return metrics, pred_ds
      else:
        return metrics

