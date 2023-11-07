import subprocess
from itertools import product

output_path = "/gpfs/users/boizardni/llm_distillation/train/models/base/pythia_410m"

param_grid = {
    'lr': [1e-6],
    'weight_decay': [0.1],
}

for params in list(product(*param_grid.values())):
    lr = params[0]
    weight_decay = params[1]
    path = output_path+f"/lr{lr}_wd{weight_decay}"

    # subprocess.call(f"mkdir {path}", shell=True)

    name = f"finetune_lr{lr}_wd{weight_decay}"
    const = "--nodes=1 --time=12:00:00 -p gpua100 --gres=gpu:1 --cpus-per-task=4"
    wrap = f"cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation; python llm-recipes/finetuning.py --model_name EleutherAI/pythia-410m-deduped --custom_dataset.file llm_distillation/dataset_generator/generated/Llama-2-7b-hf_squad_train_2s --output_dir {path} --project_name llm-distillation --batch_size_training 8 --lr {lr} --weight_decay {weight_decay}"
    subprocess.call(f"sbatch --job-name={name} {const} --wrap=\"{wrap}\"" ,shell=True)