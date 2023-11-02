import datasets

def tokenize(item, tokenizer, template):
    prompt = template.format(title=item['title'], context=item['context'], question=item['question'], answers=item['answers_generated'])

    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)
    answer_tokens = tokenizer.encode(f" {item['answers_generated']} {tokenizer.eos_token}", add_special_tokens=False)
    prompt_tokens = context_tokens+answer_tokens
    labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

    combined_tokens = {
        "input_ids": prompt_tokens,
        "labels": labels_tokens
    }

    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


def get_custom_dataset(dataset_config, tokenizer, split):

    dataset = datasets.load_from_disk(f"{dataset_config.file}/{split}")

    template = "Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"
    dataset = dataset.map(lambda item: tokenize(item, tokenizer, template), remove_columns=list(dataset.features))
    return dataset