import numpy as np
from load import tokenize
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m-deduped")
dataset = load_from_disk("dataset_generator/generated/Llama-2-7b-hf_squad_train_2s/train")

template = "Title: {title}\nContext: {context}\nQuestion: {question}\nAnswer:"
dataset = dataset.map(lambda item: tokenize(item, tokenizer, template), remove_columns=list(dataset.features))

sequence_lengths = [len(example) for example in dataset["input_ids"]]

plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(sequence_lengths, bins=50, edgecolor='k', alpha=0.7)
plt.xlabel("Sequence Length")
plt.ylabel("Number of Samples")
plt.title("Distribution of Sequence Lengths")
plt.grid(True)

mean_length = np.mean(sequence_lengths)
median_length = np.median(sequence_lengths)
q1 = np.percentile(sequence_lengths, 25)
q3 = np.percentile(sequence_lengths, 75)
min_length = min(sequence_lengths)
max_length = max(sequence_lengths)

plt.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.2f}')
plt.axvline(q1, color='green', linestyle='--', label=f'Q1: {q1:.2f}')
plt.axvline(q3, color='blue', linestyle='--', label=f'Q3: {q3:.2f}')
plt.axvline(min_length, color='purple', linestyle='--', label=f'Min: {min_length:.2f}')
plt.axvline(max_length, color='orange', linestyle='--', label=f'Max: {max_length:.2f}')
plt.legend()

plt.show()