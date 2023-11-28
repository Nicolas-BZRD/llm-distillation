import subprocess
from itertools import product

param_grid = {
    'model_name': ['EleutherAI/pythia-70m-deduped'],
    'lr': [1e-6],
    'num_epochs': [5],
    'batch_size_training': [8],
    'val_batch_size': [8],
    'weight_decay': [0.1],
    'final_div_factor': [5],

    'dataset': [
        '/gpfs/users/boizardni/llm-distillation/datasets/generator/generated/Llama-2-7b-hf_squad_train_2s_0context',
        # '/gpfs/users/boizardni/llm-distillation/datasets/loader/squad.py'
    ]
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    name_dataset = params['dataset'].split('/')[-1] if not params['dataset'].endswith('.py') else params['dataset'].split('/')[-1][:-3]
    output_path = f"/gpfs/users/boizardni/llm-distillation/train/models/distillation/{params['model_name'].split('/')[-1]}"

    subprocess.call(f"mkdir {output_path}", shell=True)

    const = "--job-name=DIST --nodes=1 --time=00:30:00 -p gpu_test --gres=gpu:1 --cpus-per-task=10 --mem-per-cpu=32G"
    pre_script = "cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} python llm-recipes/finetuning.py --distillation --model_name {params['model_name']} --lr {params['lr']} --num_epochs {params['num_epochs']} --batch_size_training {params['batch_size_training']} --val_batch_size {params['val_batch_size']} --weight_decay {params['weight_decay']} --final_div_factor {params['final_div_factor']} --custom_dataset.file {params['dataset']} --output_dir {output_path} --save_model False\""
    # print(command)
    subprocess.call(command ,shell=True)

# --project_name llm-distillation