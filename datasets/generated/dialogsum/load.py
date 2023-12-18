import os
import sys
import datasets

sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
from tools.summary.summary import *

def tokenize(item, tokenizer, pre_prompt):
    prompt = create_prompt(pre_prompt=pre_prompt, text=item['text'])

    context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
    answer_tokens = tokenizer.encode(f"{item['summary']}{tokenizer.eos_token}", add_special_tokens=False)
    prompt_tokens = context_tokens+answer_tokens
    labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

    combined_tokens = {
        "input_ids": prompt_tokens,
        "labels": labels_tokens
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_split(dataset_config, tokenizer, split):
    dataset = datasets.load_from_disk(f"{dataset_config.file}/{split}")
    pre_prompt = create_pre_prompt(context=dataset_config.context, few_shot=dataset_config.few_shot)

    dataset = dataset.map(lambda item: tokenize(item, tokenizer, pre_prompt), remove_columns=list(dataset.features))
    return dataset