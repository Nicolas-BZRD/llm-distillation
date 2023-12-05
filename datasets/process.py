from datasets import DatasetDict, load_from_disk
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Split and save a dataset")
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to the dataset to be split")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size fraction (default: 0.1%)")
    return parser.parse_args()

def main():
    args = parse_args()

    ds = load_from_disk(args.dataset_path)
    ds = ds.filter(lambda example: example['answers_generated'] is not None and example['answers_generated'] != "")
    ds = ds.train_test_split(test_size=args.val_size, seed=42)

    ds = DatasetDict({
        'train': ds['train'],
        'validation': ds['test']
    })

    ds.save_to_disk(args.dataset_path)

if __name__ == "__main__":
    main()