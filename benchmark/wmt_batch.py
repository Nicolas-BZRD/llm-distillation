import os
import json
import argparse
import sacrebleu, torch
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

def create_prompt(item, prompt_examples, prompt_template, source_column):
    if prompt_examples: item['prompt'] = prompt_examples + "\n\n" + prompt_template.format(source=item[source_column], target="")[:-1]
    else: item['prompt'] = prompt_template.format(source=item[source_column], target="")[:-1]
    return item

def tokenization(items, tokenizer):
    return tokenizer(items["prompt"], padding='longest')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for compute sacrebleu score")
    parser.add_argument("--model_id", type=str, default="EleutherAI/pythia-1b-deduped", help="Model ID")
    parser.add_argument("--data_path", type=str, default="data/wmt22/test.en-de.json", help="Path to the data file")
    parser.add_argument("--source_language", type=str, default="English", help="Source language")
    parser.add_argument("--source_column", type=str, default="en", help="Source language column")
    parser.add_argument("--target_language", type=str, default="German", help="Target language")
    parser.add_argument("--target_column", type=str, default="de", help="Target language column")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--few_shot", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Number of max new tokens")
    args = parser.parse_args()

    # GPU device
    device = get_device()

    # Tokenizer config
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = 'left'

    # Model config
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model.generation_config.max_new_tokens = 50
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # Dataset loading
    dataset = load_dataset('json', data_files=args.data_path)
    dataset = dataset['train']['translation']

    # Example prompts
    prompt_template = f"{args.source_language}: {{source}}\n{args.target_language}: {{target}}"
    prompt_examples = "\n\n".join([prompt_template.format(source=row[args.source_column], target=row[args.target_column]) for row in dataset[0:args.few_shot]])
    dataset = dataset[args.few_shot:]

    # Dataset processing
    dataset = {args.source_column: [item[args.source_column] for item in dataset], args.target_column: [item[args.target_column] for item in dataset]}
    dataset = Dataset.from_dict(dataset)
    dataset = dataset.map(lambda item: create_prompt(item, prompt_examples, prompt_template, args.source_column))
    dataset = dataset.map(lambda item: tokenization(item, tokenizer=tokenizer), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Prediction
    prompt_examples_size = len(prompt_examples) if prompt_examples else 0
    target_language_size = len(args.target_language)+2
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            generated_prediction = model.generate(batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
            generated_prediction = tokenizer.batch_decode(generated_prediction, skip_special_tokens=True)
            
            if prompt_examples:
                for item in generated_prediction:
                    predictions.append(item[prompt_examples_size:].split('\n')[3][target_language_size:])
            else:
                for item in generated_prediction:
                    predictions.append(item.split('\n')[1][target_language_size:])

    bleu = sacrebleu.corpus_bleu(predictions, dataset[args.target_column])
    print(bleu.score)
    with open(f'results/{args.model_id.split("/")[-1]}_{args.source_column}_{args.target_column}', 'w') as json_file:
        json.dump(
            {
                "source": args.source_language,
                "target": args.target_language,
                "model": args.model_id,
                "samples_number": len(predictions),
                "sacrebleau": bleu.score
            }, 
            json_file, indent=4
        )