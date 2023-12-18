import os
import sys
import datasets

sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
from tools.qa.qa import create_prompt

def tokenize(item, tokenizer):
    prompt = create_prompt(title=item['title'], context=item['context'], question=item['question'])

    context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
    answer_tokens = tokenizer.encode(f"{item['answers']['text'][0]}{tokenizer.eos_token}", add_special_tokens=False)
    prompt_tokens = context_tokens+answer_tokens
    labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

    combined_tokens = {
        "input_ids": prompt_tokens,
        "labels": labels_tokens
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_split(_, tokenizer, split):
    dataset = datasets.load_dataset("squad", split="train")

    split = "test" if split == "validation" else "train"
    dataset = dataset.train_test_split(test_size=0.1, seed=42)[split]
    dataset = dataset.map(lambda item: tokenize(item, tokenizer), remove_columns=list(dataset.features))

    return dataset