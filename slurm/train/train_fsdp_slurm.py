import subprocess
from itertools import product

param_grid = {
    'model_name': ['meta-llama/Llama-2-7b-hf'],
    'lr': [2e-5],
    'num_epochs': [3],
    'batch_size_training': [8],
    'val_batch_size': [8],
    'weight_decay': [0.1],
    'final_div_factor': [5],
    'pct_start': [0.05],

    'dataset': [
        '/gpfs/users/boizardni/llm-distillation/datasets/loader/squad.py'
    ]
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    name_dataset = params['dataset'].split('/')[-1] if not params['dataset'].endswith('.py') else params['dataset'].split('/')[-1][:-3]
    output_path = f"/gpfs/users/boizardni/llm-distillation/train/models/finetuned/{params['model_name'].split('/')[-1]}/{name_dataset}_{params['lr']}"

    subprocess.call(f"mkdir {output_path}", shell=True)

    const = "--job-name=FSDP --nodes=1 --time=24:00:00 -p gpua100 --gres=gpu:4 --cpus-per-task=10 --mem-per-cpu=12G"
    pre_script = "cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} torchrun --nnodes 1 --nproc_per_node 4 llm-recipes/finetuning.py --project_name llm-distillation --enable_fsdp --model_name {params['model_name']} --dist_checkpoint_root_folder {output_path} --custom_dataset.file {params['dataset']} --lr {params['lr']} --num_epochs {params['num_epochs']} --batch_size_training {params['batch_size_training']} --val_batch_size {params['val_batch_size']} --weight_decay {params['weight_decay']} --final_div_factor {params['final_div_factor']} --pct_start {params['pct_start']} --save_step 200 --fsdp_config.pure_bf16 --use_fast_kernels\""
    subprocess.call(command ,shell=True)