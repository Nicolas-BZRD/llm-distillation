import os
import sys
import json
import score
import torch
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from itertools import chain
from tqdm import tqdm

sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    return device

def tokenization(items, tokenizer):
    return tokenizer(items["prompt"], padding='longest')

def mapping(path, ds):
    with open(path, 'r') as f: mapping = json.load(f)
    for key, value in mapping.items():
        ds = ds.rename_column(key, value)
    return ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to benchmark a model on a dataset.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", help="Model ID")
    parser.add_argument("--model_tokenizer", type=str, help="Model tokenizer (default: model_id)")
    parser.add_argument("--dataset_id", type=str, help="Dataset hugging face ID")
    parser.add_argument("--split_name", type=str, default="test", help="Dataset split name")
    parser.add_argument("--context", action="store_true", help="To pre prompt an explanation of the task")
    parser.add_argument("--title", action="store_true", help="To keep title in the prompt")
    parser.add_argument("--number_few_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--bfloat", action="store_true", help="Load model in bf16")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions in txt file")
    parser.add_argument("--from_disk", action="store_true", help="Load dataset from disk")
    parser.add_argument("--task", type=str, default="qa", help="Benchmark type (qa, qa_generative, summarization)")
    parser.add_argument("--mapping", type=str, default="", help="JSON file to map dataset column name")
    parser.add_argument("--bert_score", action="store_true", help="To compute bert score")
    args = parser.parse_args()

    if 'chat' in args.model_id:
        from prompt.prompt import llama_chat_prompt as create_prompt

    def create_prompt_column(task, few_shot, item, has_title):
        if False: print("F here we go again but it's the last or not, so no...")
        if task == "qa" or task == "qa_generative":
            item['prompt'] = create_prompt(
                task, few_shot,
                {"title":item['title'] if has_title else "", "context":item['context'], "question":item['question']}
            )
        elif task == "qa_medical":
             item['prompt'] = create_prompt(
                task, few_shot,
                {"context":item['context'], "question":item['question']}
            )
        elif task == "summary_dialogue":
            item['prompt'] = create_prompt(
                task, few_shot,
                {"context":item['context']}
            )
        return item
    
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
    if args.bfloat and device != "cpu": model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    else: model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    logging.info('Model loaded.')

    logging.info('Processing dataset...')
    if args.from_disk:
        dataset = load_from_disk(args.dataset_id)
        if args.split_name: dataset = dataset[args.split_name]
    else: dataset = load_dataset(args.dataset_id, split=args.split_name)
    if args.mapping: dataset = mapping(args.mapping, dataset)
    has_title = True if 'title' in dataset.column_names and args.title else False
    dataset = dataset.map(lambda item: create_prompt_column(args.task, args.number_few_shot, item, has_title))
    dataset = dataset.map(lambda items: tokenization(items, tokenizer=tokenizer), batched=True, batch_size=args.batch_size)
    print(dataset['prompt'][0])
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
                max_new_tokens=80,
                do_sample=False,
                temperature=1,
                top_p=1
            )
            output = output[:, len(batch['input_ids'][0]):]
            sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
            if 'chat' in args.model_id: sentences = [item[:-4] if item.endswith("/s") else item for item in sentences]
            predictions.append([item.split('\n')[0] for item in sentences])
    logging.info('Predictions finished')

    if isinstance(dataset['answers'][0], dict): answers = [item['text'] for item in dataset['answers']]
    else: answers = dataset['answers']
    predictions = list(chain(*predictions))
    answers = answers[:len(predictions)]
    results = score.f1_score(predictions, answers)
    results['em'] = score.exact_match(predictions, answers)
    results['squad'] = (results['f1']+results['em'])/2
    results.update(score.rouge(predictions, answers))
    if args.bert_score:
        results_bert = score.bert_score(predictions, answers)
        results["f1_bert"] = sum(results_bert["f1"])/len(results_bert["f1"])
        results["precision_bert"] = sum(results_bert["precision"])/len(results_bert["precision"])
        results["recall_bert"] = sum(results_bert["recall"])/len(results_bert["recall"])
    logging.info(results)

    with open(f"{os.getenv('HOME')}/llm-distillation/benchmark/results/{args.model_id.split('/')[-1]}_{args.dataset_id.split('/')[-1]}_{args.number_few_shot}shots.json", 'w') as json_file:
        json.dump(
            {
                "model": args.model_id,
                "dataset": args.dataset_id,
                "context": args.context,
                "title": has_title,
                "number_few_shot": args.number_few_shot,
                "samples_number": len(predictions),
                **results,
            }, 
            json_file, indent=4
        )
    logging.info("Process completed.")

    if args.save_predictions:
        prediction_data = [{'answers': dataset['answers'][index], 'prediction_text': item} for index, item in enumerate(predictions)]
        with open(f"{os.getenv('HOME')}/llm-distillation/benchmark/results/predictions_{args.model_id.split('/')[-1]}_{args.dataset_id.split('/')[-1]}_{args.number_few_shot}shots.json", 'w') as file:
            for prediction_dict in prediction_data:
                json.dump(prediction_dict, file)
                file.write('\n')