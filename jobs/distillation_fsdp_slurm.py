import os
import subprocess
from itertools import product

param_grid = {
    'model_name': ['EleutherAI/pythia-410m-deduped', 'bigscience/bloomz-560m', 'facebook/opt-350m'],
    'teacher_model_name': ['tiiuae/falcon-7b-instruct', 'mistralai/Mistral-7B-Instruct-v0.2', 'meta-llama/Llama-2-7b-chat-hf'],
    'lr': [2e-6],
    'num_epochs': [5],
    'batch_size_training': [8],
    'val_batch_size': [8],
    'distil_factor': [0, 1.5],

    'dataset': [f"{os.getenv('HOME')}/llm-distillation/datasets/loader/qed.py"]
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
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

    const = "--job-name=Distillation --nodes=1 --time=01:00:00 -p gpu_test --gres=gpu:4 --cpus-per-task=10 --mem-per-cpu=32G"
    pre_script = "cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} torchrun --nnodes 1 --nproc_per_node 4 --master_port 29510 llm-recipes/finetuning.py --model_name {params['model_name']} --dataset.file {params['dataset']} --lr {params['lr']} --num_epochs {params['num_epochs']} --batch_size_training {params['batch_size_training']} --val_batch_size {params['val_batch_size']} --output_dir {output_path} --distillation_config.model_name {params['teacher_model_name']} --distillation --distillation_config.enable_fsdp --distillation_config.pure_bf16 --distillation_config.distil_factor {params['distil_factor']} --save_step 150 --save_all\""
    subprocess.call(command ,shell=True)
