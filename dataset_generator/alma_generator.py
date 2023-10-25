import os
import csv
import torch
import logging
import warnings
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    return device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate parallel data")
    parser.add_argument("--model_id", type=str, default="haoranxu/ALMA-7B", help="Model ID")
    parser.add_argument("--data_path", type=str, default="data/parallel_data/train.de-en.json", help="Path to the data file or hub id")
    parser.add_argument("--source_language", type=str, default="English", help="Source language")
    parser.add_argument("--source_column", type=str, default="en", help="Source language column")
    parser.add_argument("--target_language", type=str, default="German", help="Target language")
    parser.add_argument("--target_column", type=str, default="de", help="Target language column")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of data loader workers")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of max new tokens")
    args = parser.parse_args()

    # GPU device
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.info('Start')
    device = get_device()
    logging.info(f'Device: {device}')

    # Tokenizer config
    logging.info(f'Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side='left')
    logging.info(f'Tokenizer loaded.')

    # Model config
    logging.info('Loading model...')
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map="auto")
    model.generation_config.num_beams=5
    model.generation_config.max_new_tokens=args.max_new_tokens
    model.generation_config.do_sample=True
    model.generation_config.temperature=0.6
    model.generation_config.top_p=0.9
    model.to(device)
    logging.info('Model loaded.')

    # Dataset loading
    logging.info('Processing dataset...')
    prompt_template = f"Translate this from {args.source_language} to {args.target_language}:\n{args.source_language}: {{source}}\n{args.target_language}: "
    if(args.data_path.endswith(".json")):
        dataset = load_dataset('json', data_files=args.data_path)
        dataset = Dataset.from_dict({
            args.source_column: [item[args.source_column] for item in dataset['train']['translation']],
            args.target_column: [item[args.target_column] for item in dataset['train']['translation']]
        })
        dataset = dataset.map(lambda item: {'prompt': prompt_template.format(source=item[args.source_column])})
        dataset = dataset.map(lambda items: tokenizer(items["prompt"], padding='longest'), batched=True, batch_size=args.batch_size)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        print("TODO")
        exit()
    logging.info('Dataset processed...')

    # Create output CSV file if needed
    csv_path = f'data/generated/{args.model_id.split("/")[-1]}_{args.source_column}_{args.target_column}.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([args.source_column, args.target_column])

    # Prediction
    logging.info('Starting predictions...')
    source_language_size = len(args.source_language)+2
    target_language_size = len(args.target_language)+2
    sources = []
    predictions = []
    batch_number = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_number+=1
            generated_prediction = model.generate(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), pad_token_id=tokenizer.pad_token_id)
            generated_prediction = tokenizer.batch_decode(generated_prediction, skip_special_tokens=True)
            
            for item in generated_prediction:
                item = item.split('\n')
                sources.append(item[1][source_language_size:])
                predictions.append(item[2][target_language_size:])

            if batch_number % 2 == 0:
                with open(csv_path, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for source, prediction in zip(sources, predictions):
                        csv_writer.writerow([source, prediction])
                sources = []
                predictions = []
                batch_number = 0
    logging.info('Predictions finished')