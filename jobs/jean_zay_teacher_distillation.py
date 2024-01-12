import os
import subprocess
from itertools import product

param_grid = {
    'model_name': ['EleutherAI/pythia-410m-deduped', 'bigscience/bloomz-560m', 'facebook/opt-350m'],
    'teacher_model_name': ['tiiuae/falcon-7b-instruct', 'mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Llama-2-7b-chat-hf'],
    'distil_factor': [0, 1.5],

    'dataset': [f"{os.getenv('HOME')}/llm-distillation/datasets/loader/squad.py"]
}
param_names = param_grid.keys()

time = "14:00:00"
gpu=4
save_step = 150
lr=2e-6
batch = 8
partition = 'gpu_p5'

i = 0
for param_values in product(*param_grid.values()):
    i+=1
    ip=29000+i
    params = dict(zip(param_names, param_values))
    name_dataset = params['dataset'].split('/')[-1] if not params['dataset'].endswith('.py') else params['dataset'].split('/')[-1][:-3]

    output_path = os.path.join(
        os.getenv('HOME'),
        'llm-distillation',
        'train',
        params['teacher_model_name'].split('/')[-1],
        params['model_name'].split('/')[-1],
        name_dataset,
        f'd{params["distil_factor"]}'
    )
    os.makedirs(output_path, exist_ok=True)
    const = f"#!/bin/bash \n\n#SBATCH --job-name=D_{name_dataset} \n#SBATCH --nodes=1 \n#SBATCH --time={time} \n#SBATCH --account=dgo@a100 \n#SBATCH -C a100 \n#SBATCH --partition={partition} \n#SBATCH --gres=gpu:{gpu} \n#SBATCH --ntasks=4 \n#SBATCH --cpus-per-task=10\n\n"
    post_command = f"cd {os.getenv('HOME')}\nmodule load anaconda-py3/2023.09 \nsource activate llm_distillation \n\nexport TRANSFORMERS_OFFLINE=1 \nexport TOKENIZERS_PARALLELISM=true \nexport HF_DATASET_OFFLINE=1 \n\nexport WANDB_MODE=offline \nexport WANDB_DIR={os.getenv('HOME')}/llm-distillation/train \nexport WANDB_CACHE_DIR={os.getenv('HOME')}/llm-distillation/train \nexport WANDB_CONFIG_DIR={os.getenv('HOME')}/llm-distillation/train\n\n"
    command = f"{const}{post_command}torchrun --nnodes 1 --nproc_per_node {gpu} --master_port {ip} llm-recipes/finetuning.py --model_name {params['model_name']} --dataset.file {params['dataset']} --lr {lr} --num_epochs 5 --batch_size_training {batch} --val_batch_size {batch} --output_dir {output_path} --distillation_config.model_name {params['teacher_model_name']} --distillation --distillation_config.enable_fsdp --distillation_config.pure_bf16 --distillation_config.distil_factor {params['distil_factor']} --save_step {save_step} --save_all"

    with open("slurm_tmp.sh", "w") as f:
        f.write(command)

    subprocess.call(f"sbatch slurm_tmp.sh",shell=True)
    subprocess.call(f"rm slurm_tmp.sh",shell=True)