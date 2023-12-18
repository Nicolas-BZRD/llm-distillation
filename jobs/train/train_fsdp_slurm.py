import subprocess
from itertools import product

param_grid = {
    'model_name': ['meta-llama/Llama-2-7b-hf'],
    'lr': [1e-4],
    'num_epochs': [1],
    'batch_size_training': [16],
    'val_batch_size': [16],
    'final_div_factor': [5],
    'dataset': [
        '/gpfs/users/boizardni/llm-distillation/datasets/loader/squad.py'
    ]
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    name_dataset = params['dataset'].split('/')[-1] if not params['dataset'].endswith('.py') else params['dataset'].split('/')[-1][:-3]
    output_path = f"/gpfs/users/boizardni/llm-distillation/train/models/finetuned/{params['model_name'].split('/')[-1]}/{name_dataset}_peft"

    subprocess.call(f"mkdir {output_path}", shell=True)

    const = "--job-name=train_fsdp --nodes=1 --time=02:00:00 -p gpua100 --gres=gpu:4 --cpus-per-task=10 --mem-per-cpu=12G"
    pre_script = "cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} torchrun --nnodes 1 --nproc_per_node 4 llm-recipes/finetuning.py --project_name test --enable_fsdp --model_name {params['model_name']} --output_dir {output_path} --dataset.file {params['dataset']} --lr {params['lr']} --num_epochs {params['num_epochs']} --batch_size_training {params['batch_size_training']} --val_batch_size {params['val_batch_size']} --final_div_factor {params['final_div_factor']} --pure_bf16 --save_step 100\""
    subprocess.call(command ,shell=True)