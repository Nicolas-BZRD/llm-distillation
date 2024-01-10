import os
import subprocess

runs = [
    {"model": "meta-llama/Llama-2-7b-chat-hf", "few_shot": 5, "title": False, "dataset": "/gpfs/users/boizardni/llm-distillation/datasets/processed/qed"},
    {"model": "mistralai/Mistral-7B-Instruct-v0.2", "few_shot": 5, "title": True, "dataset": "/gpfs/users/boizardni/llm-distillation/datasets/processed/qed"},
    {"model": "tiiuae/falcon-7b-instruct", "few_shot": 3, "title": False, "dataset": "/gpfs/users/boizardni/llm-distillation/datasets/processed/qed"},

    {"model": "meta-llama/Llama-2-7b-chat-hf", "few_shot": 2, "title": False, "dataset": "GEM/FairytaleQA"},
    {"model": "mistralai/Mistral-7B-Instruct-v0.2", "few_shot": 4, "title": False, "dataset": "GEM/FairytaleQA"},
    {"model": "tiiuae/falcon-7b-instruct", "few_shot": 2, "title": False, "dataset": "GEM/FairytaleQA"},

    {"model": "meta-llama/Llama-2-7b-chat-hf", "few_shot": 3, "title": False, "dataset": "/gpfs/users/boizardni/llm-distillation/datasets/processed/dialogsum"},
    {"model": "mistralai/Mistral-7B-Instruct-v0.2", "few_shot": 2, "title": False, "dataset": "/gpfs/users/boizardni/llm-distillation/datasets/processed/dialogsum"},
    {"model": "tiiuae/falcon-7b-instruct", "few_shot": 2, "title": False, "dataset": "/gpfs/users/boizardni/llm-distillation/datasets/processed/dialogsum"},

    {"model": "meta-llama/Llama-2-7b-chat-hf", "few_shot": 1, "title": False, "dataset": "squad"},
    {"model": "mistralai/Mistral-7B-Instruct-v0.2", "few_shot": 3, "title": True, "dataset": "squad"},
    {"model": "tiiuae/falcon-7b-instruct", "few_shot": 4, "title": False, "dataset": "squad"},

    {"model": "meta-llama/Llama-2-7b-chat-hf", "few_shot": 3, "title": False, "dataset": "/gpfs/users/boizardni/llm-distillation/datasets/processed/pubmed_qa_50k"},
    {"model": "mistralai/Mistral-7B-Instruct-v0.2", "few_shot": 3, "title": True, "dataset": "/gpfs/users/boizardni/llm-distillation/datasets/processed/pubmed_qa_50k"},
    {"model": "tiiuae/falcon-7b-instruct", "few_shot": 3, "title": False, "dataset": "/gpfs/users/boizardni/llm-distillation/datasets/processed/pubmed_qa_50k"},
]

for run in runs:
    const = "--job-name=generator --nodes=1 --time=24:00:00 -p gpua100 --gres=gpu:1 --cpus-per-task=8 --mem-per-cpu=32G"
    pre_script = "cd /gpfs/users/boizardni/llm-distillation; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"

    subprocess.call(f"mkdir {os.getenv('HOME')}/llm-distillation/datasets/generated/{run['model'].split('/')[-1]}" ,shell=True)
    subprocess.call(f"mkdir {os.getenv('HOME')}/llm-distillation/datasets/generated/{run['model'].split('/')[-1]}/{run['dataset'].split('/')[-1]}" ,shell=True)

    command = f"sbatch {const} --wrap=\"{pre_script} python datasets/generator.py --model_id {run['model']} --dataset_id {run['dataset']} --number_few_shot {run['few_shot']} --bfloat"

    # title
    if run['title']: command += " --title"

    # from disk
    if "llm-distillation/datasets" in run['dataset']: command += " --from_disk"

    # mapping
    if "dialogsum" in run['dataset'] or "qed" in run['dataset'] or "FairytaleQA" in run['dataset']: command += f" --mapping {os.getenv('HOME')}/llm-distillation/benchmark/mapping/{run['dataset'].split('/')[-1]}.json"

    # mapping dict
    if "qed" in run['dataset']: command += " --mapping_dict string"

    # batch_size, task and split
    splits = ['train']
    if "qed" in run['dataset'] or "squad" in run['dataset']: command += " --batch_size 4 --task qa"
    elif "FairytaleQA" in run['dataset']:
        command += " --batch_size 4 --task qa_generative"
        splits.append("validation")
    elif "dialogsum" in run['dataset']:
        command += " --batch_size 4 --task summary_dialogue"
        splits.append("validation")
    elif "pubmed_qa" in run['dataset']:
        command += " --batch_size 4 --task qa_medical"
        splits.append("validation")

    for split in splits:
        subprocess.call(command + f" --split_name {split}\"" ,shell=True)