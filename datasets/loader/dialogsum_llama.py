import os
import sys
from datasets import load_dataset, load_from_disk

sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
from prompt.prompt import llama_chat_prompt
from prompt.prompt import create_prompt

def tokenize(item, tokenizer):
    if 'chat' in tokenizer.name_or_path:
        prompt = llama_chat_prompt(
            'summary_dialogue', 3,
            {"context":item['context']}
        )
    else:
        prompt = create_prompt(
            'summary_dialogue', 3 if 'llama' in tokenizer.name_or_path else 0,
            {"context":item['context']}
        )
    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
    
    if 'llama' in tokenizer.name_or_path: answer_tokens = tokenizer.encode(f"{item['summary_generated']}", add_special_tokens=False)
    else: answer_tokens = tokenizer.encode(f" {item['summary_generated']}{tokenizer.eos_token}", add_special_tokens=False)

    prompt_tokens = context_tokens+answer_tokens
    labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

    combined_tokens = {
        "input_ids": prompt_tokens,
        "labels": labels_tokens
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_split(_, tokenizer, split):
    # dataset = load_dataset("Nicolas-BZRD/dialogsum_llama", split=split)
    dataset = load_from_disk(f"{os.getenv('HOME')}/llm-distillation/datasets/llama_generated/dialogsum")
    dataset = dataset[split]
    dataset = dataset.map(lambda item: tokenize(item, tokenizer), remove_columns=list(dataset.features))
    return dataset