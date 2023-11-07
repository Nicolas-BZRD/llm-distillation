import os
import json
import score
import torch
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset
from itertools import chain
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"

contexts = [
    "Reading comprehension with the answer to the question to be extracted from the passage only.\n\n",
    "Extract the text that answers the question. If the answer is not in the passage, answer 'no response'.\n\n"
]

templates = [
    "Title: {title}\nPassage: {context}\nQuestion: {question}\nAnswer: {answers}",
    "Passage: {context}\nQuestion: {question}\nAnswer: {answers}",
]

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    return device

def create_few_shot(number_few_shot, has_title):
    with open('prompt_examples.json') as json_file:
        data = json.load(json_file)

    prompt = "\n\n".join([
        (templates[0].format(title=row['title'], context=row['context'], question=row['question'], answers=row['answers'])
        if has_title else templates[1].format(context=row['context'], question=row['question'], answers=row['answers']))
        for row in data[0:number_few_shot]
    ]) + '\n\n'
    return prompt


def create_prompt(item, pre_prompt, has_title):
    if has_title: prompt = templates[0].format(title=item['title'], context=item['context'], question=item['question'], answers="")[:-1]
    else: prompt = templates[1].format(context=item['context'], question=item['question'], answers="")[:1]
    item['prompt'] = pre_prompt+prompt
    return item

def tokenization(items, tokenizer):
    return tokenizer(items["prompt"], padding='longest')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for compute sacrebleu score")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", help="Model ID")
    parser.add_argument("--model_tokenizer", type=str, help="Model tokenizer (default: model_id)")
    parser.add_argument("--dataset_id", type=str, default="squad", help="Dataset hugging face ID")
    parser.add_argument("--split_name", type=str, default="validation", help="Dataset split name")
    parser.add_argument("--context_id", type=int, default=-1, help="Id of the context to use (default: -1 no context)")
    parser.add_argument("--number_few_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions in txt file")
    parser.add_argument("--sample", action="store_true", help="Process on a sample of 1000 elements")
    args = parser.parse_args()
    
    
    logging.basicConfig(level=logging.INFO)
    logging.info('Start')
    device = get_device()
    logging.info(f'Device: {device}')

    logging.info(f'Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer if args.model_tokenizer else args.model_id)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.padding_side = 'left'
    logging.info(f'Tokenizer loaded.')

    logging.info('Loading model...')
    if args.model_id.startswith('meta-llama') and device != "cpu":
        if device.type == "cuda": # Bug with cuda: <unk> tokens in float16 and not in bfloat 16
            model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
        else: # Work with mps
            model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    logging.info('Model loaded.')

    logging.info('Processing dataset...')
    dataset = load_dataset(args.dataset_id, split=args.split_name) if not args.sample else load_dataset(args.dataset_id, split=args.split_name+"[0:1000]")
    has_title = True if 'title' in dataset.column_names else False
    pre_prompt = "" if args.context_id == -1 else contexts[args.context_id]
    pre_prompt += create_few_shot(args.number_few_shot, has_title) if args.number_few_shot>0 else ""
    dataset = dataset.map(lambda item: create_prompt(item, pre_prompt, has_title))
    dataset = dataset.map(lambda items: tokenization(items, tokenizer=tokenizer), batched=True, batch_size=args.batch_size)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    logging.info('Dataset processed...')

    logging.info('Starting predictions...')
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            output = model.generate(
                batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                max_new_tokens=20,
                do_sample=False,
                temperature=1,
                top_p=1
            )
            output = output[:, len(batch['input_ids'][0]):]
            sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
            predictions.append([item.split('\n')[0] for item in sentences])
    logging.info('Predictions finished')

    answers = [item['text'] for item in dataset['answers']]
    predictions = list(chain(*predictions))
    results = score.f1_score(predictions, answers)
    results['em'] = score.exact_match(predictions, answers)
    results['squad'] = (results['f1']+results['em'])/2
    logging.info(results)

    with open(f'results/{args.model_id.split("/")[-1]}_{args.dataset_id}_{args.number_few_shot}s_{args.context_id}context.json', 'w') as json_file:
        json.dump(
            {
                "model": args.model_id,
                "number_few_shot": args.number_few_shot,
                "context": "" if args.context_id == -1 else contexts[args.context_id],
                "dataset": args.dataset_id,
                "samples_number": len(predictions),
                "f1": results['f1'],
                "precision": results['precision'],
                "recall": results['recall'],
                "em": results['em'],
                "squad": results['squad']
            }, 
            json_file, indent=4
        )
    logging.info("Process completed.")

if args.save_predictions:
    prediction_data = [{'id': dataset['id'][index], 'prediction_text': item} for index, item in enumerate(predictions)]
    with open(f"results/predictions.json", 'w') as file:
        for prediction_dict in prediction_data:
            json.dump(prediction_dict, file)
            file.write('\n')