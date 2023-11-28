import subprocess
from itertools import product

param_grid = {
    'model_name': ['EleutherAI/pythia-410m-deduped'],
    'lr': [1e-6],
    'num_epochs': [3],
    'batch_size_training': [4],
    'val_batch_size': [4],
    'weight_decay': [0.1],
    'final_div_factor': [5],
    'pct_start': [0.05],
    'distil_factor': [0, 0.5, 1, 1.5],

    'dataset': [
        '/gpfs/users/boizardni/llm-distillation/datasets/generator/generated/Llama-2-7b-hf_squad_train_2s_0context',
    ]
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    name_dataset = params['dataset'].split('/')[-1] if not params['dataset'].endswith('.py') else params['dataset'].split('/')[-1][:-3]
    output_path = f"/gpfs/users/boizardni/llm-distillation/train/models/distillation/{params['model_name'].split('/')[-1]}/{name_dataset}_{params['distil_factor']}"

    subprocess.call(f"mkdir {output_path}", shell=True)

    const = "--job-name=DIST_FSDP --nodes=1 --time=04:00:00 -p gpua100 --gres=gpu:4 --cpus-per-task=10 --mem-per-cpu=32G"
    pre_script = "cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} torchrun --nnodes 1 --nproc_per_node 4 llm-recipes/finetuning.py --project_name llm-distillation --model_name {params['model_name']} --pure_bf16 --dataset.file {params['dataset']} --lr {params['lr']} --num_epochs {params['num_epochs']} --batch_size_training {params['batch_size_training']} --val_batch_size {params['val_batch_size']} --weight_decay {params['weight_decay']} --final_div_factor {params['final_div_factor']} --output_dir {output_path} --pct_start {params['pct_start']} --distillation --distillation_config.enable_fsdp --distillation_config.pure_bf16 --distillation_config.distil_factor {params['distil_factor']} --distillation_config.use_fast_kernels --distillation_config.context --distillation_config.few_shot 2 --save_step 600\""
    print(command)
    subprocess.call(command ,shell=True)