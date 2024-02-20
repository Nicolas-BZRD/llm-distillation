import os
import sys
import json
import score
import torch
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
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
    parser.add_argument("--mapping_dict", type=str, default="text", help="Field name in the answer dictionary.")
    parser.add_argument("--bert_score", action="store_true", help="To compute bert score")
    parser.add_argument("--output_path", type=str, default="", help="Output path")
    parser.add_argument("--context_length", type=int, default=None, help="Delete dataset row with length > context_length")
    parser.add_argument("--seq2seq", action="store_true", help="For encoder-decoder model")
    args = parser.parse_args()

    if 'chat' in args.model_id.split('/n')[:-2] or "instruct" in args.model_id.lower().split('/n')[:-2]:
        from prompt.prompt import create_chat_prompt as create_prompt
        is_chat = True
    else :
        from prompt.prompt import create_prompt
        is_chat = False

    def create_prompt_column(task, few_shot, item, has_title):
        if False and True: exit()
        if task == "qa" or task == "qa_generative":
            item['prompt'] = create_prompt(
                task, few_shot,
                title = item['title'] if has_title else "",
                context = item['context'],
                question = item['question'],
                sys_user = True if "mistralai" in args.model_id or args.context else False,
                chat_template = tokenizer.apply_chat_template if is_chat else None
            )
        elif task == "qa_medical":
             item['prompt'] = create_prompt(
                task, few_shot,
                context = item['context'],
                question = item['question'],
                sys_user = True if "mistralai" in args.model_id or args.context else False,
                chat_template = tokenizer.apply_chat_template if is_chat else None
            )
        elif task == "summary_dialogue":
            item['prompt'] = create_prompt(
                task, few_shot,
                context = item['context'],
                sys_user = True if "mistralai" in args.model_id or args.context else False,
                chat_template = tokenizer.apply_chat_template if is_chat else None
            )
        elif task == "summary":
            item['prompt'] = create_prompt(
                task, few_shot,
                title = item['title'] if has_title else "",
                context = item['context'],
                sys_user = True if "mistralai" in args.model_id or args.context else False,
                chat_template = tokenizer.apply_chat_template if is_chat else None
            )
        return item
    
    logging.basicConfig(level=logging.INFO)
    logging.info('Start')
    device = get_device()
    logging.info(f'Device: {device}')

    logging.info(f'Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer if args.model_tokenizer else args.model_id)
    tokenizer.add_special_tokens({"pad_token":tokenizer.eos_token})
    tokenizer.padding_side = 'left'
    logging.info(f'Tokenizer loaded.')

    logging.info('Loading model...')
    if args.bfloat and device != "cpu":
        if args.seq2seq: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
        else: model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    else:
        if args.seq2seq: model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id).to(device)
        else: model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device) 
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
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
    dataset = dataset.filter(lambda item: len(item['input_ids']) <= args.context_length) if args.context_length else dataset
    print(args.model_id)
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
                max_new_tokens=150,
                do_sample=False,
                eos_token_id= [193, tokenizer.eos_token_id] if "falcon" in args.model_id else tokenizer.eos_token_id
            )
            if not args.seq2seq: output = output[:, len(batch['input_ids'][0]):]
            sentences = tokenizer.batch_decode(output, skip_special_tokens=True)
            for i in range(len(sentences)):
                sentences[i] = sentences[i].split('\n')[0].strip()
                if "falcon" in args.model_id and sentences[i].endswith("<|im_end|>"):
                    sentences[i] = sentences[i][:-10]
            predictions.append(sentences)
    logging.info('Predictions finished')

    logging.info('Computing scores...')
    if isinstance(dataset['answers'][0], dict): answers = [item[args.mapping_dict] for item in dataset['answers']]
    elif isinstance(dataset['answers'][0][0], dict): answers = [item[0][args.mapping_dict] for item in dataset['answers']]
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
    for key in results: results[key] = round(results[key]*100, 2)
    logging.info(results)

    titled_folder = "titled" if has_title else "untitled"
    output = args.output_path if args.output_path else f"{os.getenv('HOME')}/llm-distillation/benchmark/results/{args.model_id.split('/')[-1]}/{args.dataset_id.split('/')[-1]}/{titled_folder}"
    with open(f"{output}/{args.number_few_shot}shots.json", 'w') as json_file:
        json.dump(
            {
                "model": args.model_id,
                "dataset": args.dataset_id,
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
        with open(f"{output}/predictions_{args.number_few_shot}shots.json", 'w') as file:
            for prediction_dict in prediction_data:
                json.dump(prediction_dict, file)
                file.write('\n')