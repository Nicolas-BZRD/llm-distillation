import os
import sys
from datasets import load_from_disk

sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
from prompt.prompt import create_chat_prompt
from prompt.prompt import create_prompt

def tokenize(item, tokenizer):
    is_chat = True if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower() else False
    task = "qa_generative"

    if tokenizer.name_or_path == "meta-llama/Llama-2-7b-chat-hf":
        shot = 2
    elif tokenizer.name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
        shot = 4
    elif tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
        shot = 2

    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            context = item['context'],
            question = item['question'],
            sys_user = True if "mistralai/Mistral-7B-Instruct-v0.2" in tokenizer.name_or_path else False,
            chat_template = tokenizer.apply_chat_template
        )
    else:
        prompt = create_prompt(
            task, 0, 
            context = item['context'],
            question = item['question'],
        )

    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
    
    if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower():
        context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
        if tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
            answer_tokens = tokenizer.encode(f" {item['answers_generated']}", add_special_tokens=False)
        else:
            answer_tokens = tokenizer.encode(f"{item['answers_generated']}", add_special_tokens=False)
    else:
        context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
        answer_tokens = tokenizer.encode(f" {item['answers_generated']}{tokenizer.eos_token}", add_special_tokens=False)

    prompt_tokens = context_tokens+answer_tokens
    labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

    combined_tokens = {
        "input_ids": prompt_tokens,
        "labels": labels_tokens
    }
    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_split(dataset_config, tokenizer, split):
    dataset = load_from_disk(f"{os.getenv('HOME')}/llm-distillation/datasets/hf/{dataset_config.generated_by.split('/')[-1]}-FairytaleQA")
    dataset = dataset[split]
    if dataset_config.size < 1: dataset = dataset.select(range(int(len(dataset)*dataset_config.size)))
    dataset = dataset.map(lambda item: tokenize(item, tokenizer), remove_columns=list(dataset.features))
    return dataset