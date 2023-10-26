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

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    return device

def create_few_shot(number_few_shot):
    with open('prompt_examples.json') as json_file:
        data = json.load(json_file)

    template = "Context: {context}\nQuestion: {question}\nAnswer:{answers}"
    prompt = "\n\n".join([template.format(
        context=row['context'],
        question=row['question'],
        answers=row['answers']
    ) for row in data[0:number_few_shot]])
    return prompt+'\n\n'

def create_prompt(item, prompt_examples):
    template = "Context: {context}\nQuestion: {question}\nAnswer:"
    prompt = template.format(context=item['context'], question=item['question'])
    if prompt_examples:
        item['prompt'] = prompt_examples+prompt
    else:
        item['prompt'] = prompt
    return item

def tokenization(items, tokenizer):
    return tokenizer(items["prompt"], padding='longest')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for compute sacrebleu score")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", help="Model ID")
    parser.add_argument("--dataset_id", type=str, default="squad", help="Dataset hugging face ID")
    parser.add_argument("--split_name", type=str, default="validation", help="Dataset split name")
    parser.add_argument("--number_few_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of data loader workers")
    args = parser.parse_args()
    
    
    logging.basicConfig(level=logging.INFO)
    logging.info('Start')
    device = get_device()
    logging.info(f'Device: {device}')

    logging.info(f'Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    tokenizer.padding_side = 'left'
    logging.info(f'Tokenizer loaded.')

    logging.info('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.do_sample = False
    model.generation_config.temperature = 1
    model.generation_config.top_p = 1
    logging.info('Model loaded.')

    logging.info('Processing dataset...')
    dataset = load_dataset(args.dataset_id, split=args.split_name)
    prompt_examples = create_few_shot(args.number_few_shot) if args.number_few_shot>0 else ""
    prompt_examples_length = len(prompt_examples)
    dataset = dataset.map(lambda item: create_prompt(item, prompt_examples))
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
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=15,
                do_sample=False,
                num_beams=1
            ).to('cpu')
            sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
            predictions.append([item[prompt_examples_length:].split('\n')[2][8:] for item in sentences])
    logging.info('Predictions finished')
    
    answers = [item['text'] for item in dataset['answers']]
    predictions = list(chain(*predictions))
    results = score.f1_score(predictions, answers)
    results['em'] = score.exact_match(predictions, answers)
    results['squad'] = (results['f1']+results['em'])/2
    logging.info(results)

    with open(f'results/{args.model_id.split("/")[-1]}_{args.dataset_id}.json', 'w') as json_file:
        json.dump(
            {
                "model": args.model_id,
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