import argparse
import json
from datasets import load_dataset, load_metric

def main(args):
    squad_metric = load_metric("squad")

    ds = load_dataset(args.dataset, split=args.split)
    references = [{"id": item['id'], "answers": item['answers']} for item in ds]

    with open(args.predictions_file, 'r') as file:
        predictions = [json.loads(line) for line in file]

    results = squad_metric.compute(predictions=predictions, references=references)
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SQuAD predictions")
    parser.add_argument("--dataset", required=True, help="Name of the dataset (e.g., 'squad')")
    parser.add_argument("--split", required=True, help="Name of the split (e.g., 'validation')")
    parser.add_argument("--predictions_file", required=True, help="Path to the predictions file")

    args = parser.parse_args()
    main(args)
