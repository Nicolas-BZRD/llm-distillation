import os
import subprocess

def find_directory(number, directory_list):
    directory_list = [int(d) for d in directory_list]
    directory_list.sort()
    
    directory_index = directory_list[0]

    for dir in directory_list:
        if dir < number:
            directory_index = dir
        else:
            break
        
    return str(directory_index)

partition = 'gpu_p4'
time = "01:00:00"
batch = 4
gpu=1

model_name = 'facebook/opt-350m'
dataset = f"{os.getenv('HOME')}/llm-distillation/datasets/processed/qed"
split="validation"
task="qa"

checkpoint = [
    ['falcon-7b-instruct', "d0", [886]],
    ['Mistral-7B-Instruct-v0.2', "d0",[1931]],
    ['Llama-2-7b-chat-hf', "d0", [2218]],

    ['falcon-7b-instruct', "d1.5", [2003, 2218]],
    ['Mistral-7B-Instruct-v0.2', "d1.5", [1931, 2223]],
    ['Llama-2-7b-chat-hf', "d1.5", [2218, 1927]],
]

for row in checkpoint:
    checkpoint_path = f"{os.getenv('HOME')}/llm-distillation/train/{row[0]}/{model_name.split('/')[-1]}/{dataset.split('/')[-1]}/{row[1]}"
    folder = os.listdir(checkpoint_path)
    for checkpoint in row[2]:
        index = find_directory(checkpoint, folder)

        output_path = os.path.join(
            os.getenv('HOME'),
            'llm-distillation',
            'benchmark',
            'results',
            model_name.split('/')[-1],
            dataset.split('/')[-1],
            row[0],
            row[1]
        )

        os.makedirs(output_path, exist_ok=True)

        const = f"#!/bin/bash \n\n#SBATCH --job-name=B_{dataset} \n#SBATCH --nodes=1 \n#SBATCH --time={time} \n#SBATCH --account=dgo@a100 \n#SBATCH -C a100 \n#SBATCH --partition={partition} \n#SBATCH --gres=gpu:{gpu} \n#SBATCH --ntasks=4 \n#SBATCH --cpus-per-task=10\n\n"
        post_command = f"cd {os.getenv('HOME')}\nmodule load anaconda-py3/2023.09 \nsource activate llm_distillation \n\nexport TRANSFORMERS_OFFLINE=1 \nexport TOKENIZERS_PARALLELISM=true \nexport HF_DATASET_OFFLINE=1\n\n"
        command = f"{const}{post_command}python llm-distillation/benchmark/benchmark.py --model_id {checkpoint_path+'/'+index} --model_tokenizer {model_name} --dataset {dataset} --split_name {split} --batch_size {batch} --number_few_shot 0 --task {task} --bert_score --save_predictions ----output_path {output_path}"

        # Mapping column name
        if dataset in [f"{os.getenv('HOME')}/llm-distillation/datasets/processed/dialogsum", 'GEM/FairytaleQA', f"{os.getenv('HOME')}/llm-distillation/datasets/processed/qed"]:
            mapping = f"{os.getenv('HOME')}/llm-distillation/benchmark/mapping/{dataset.split('/')[-1]}.json"
            command += f" --mapping {mapping}"

        # Local dataset
        if dataset in [f"{os.getenv('HOME')}/llm-distillation/datasets/processed/dialogsum", f"{os.getenv('HOME')}/llm-distillation/datasets/processed/pubmed_qa", f"{os.getenv('HOME')}/llm-distillation/datasets/processed/qed"]:
            command += " --from_disk"

        # Mapping dict
        if dataset in [f"{os.getenv('HOME')}/llm-distillation/datasets/processed/qed"]:
            command += " --mapping_dict string"
        
        with open("slurm_tmp.sh", "w") as f:
            f.write(command)

        subprocess.call(f"sbatch slurm_tmp.sh",shell=True)
        subprocess.call(f"rm slurm_tmp.sh",shell=True)