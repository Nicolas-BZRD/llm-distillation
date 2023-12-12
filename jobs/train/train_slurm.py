import subprocess
from itertools import product

param_grid = {
    'model_name': ['EleutherAI/pythia-410m-deduped'],
    'lr': [1e-6],
    'num_epochs': [5],
    'batch_size_training': [2],
    'val_batch_size': [2],
    'weight_decay': [0.1],
    'final_div_factor': [5],

    'dataset': ['/gpfs/users/boizardni/llm-distillation/datasets/generated/qed']
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    name_dataset = params['dataset'].split('/')[-1] if not params['dataset'].endswith('.py') else params['dataset'].split('/')[-1][:-3]
    output_path = f"/gpfs/users/boizardni/llm-distillation/train/models/finetuned/{params['model_name'].split('/')[-1]}/{name_dataset}_{params['lr']}"

    subprocess.call(f"mkdir {output_path}", shell=True)

    const = "--job-name=train --nodes=1 --time=05:00:00 -p gpua100 --gres=gpu:1 --cpus-per-task=4"
    pre_script = "cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} python llm-recipes/finetuning.py --project_name llm-distillation-qed --model_name {params['model_name']} --lr {params['lr']} --num_epochs {params['num_epochs']} --batch_size_training {params['batch_size_training']} --val_batch_size {params['val_batch_size']} --weight_decay {params['weight_decay']} --final_div_factor {params['final_div_factor']} --dataset.file {params['dataset']} --output_dir {output_path} --save_step 300\""
    subprocess.call(command ,shell=True)