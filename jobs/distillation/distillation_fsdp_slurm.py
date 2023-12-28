import subprocess
from itertools import product

param_grid = {
    'model_name': ['EleutherAI/pythia-410m-deduped'],
    'teacher_model_name': ['meta-llama/Llama-2-7b-chat-hf'],
    'lr': [1e-6],
    'num_epochs': [5],
    'batch_size_training': [2],
    'val_batch_size': [2],
    'final_div_factor': [5],
    'distil_factor': [1],
    'temperature': [1],

    'dataset': ['/gpfs/users/boizardni/llm-distillation/datasets/loader/dialogsum_llama.py']
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    name_dataset = params['dataset'].split('/')[-1] if not params['dataset'].endswith('.py') else params['dataset'].split('/')[-1][:-3]
    output_path = f"/gpfs/users/boizardni/llm-distillation/train/models/distillation/{params['model_name'].split('/')[-1]}_dist{params['distil_factor']}_temp{params['temperature']}_lr{params['lr']}"

    subprocess.call(f"mkdir {output_path}", shell=True)

    const = "--job-name=DIST_FSDP --nodes=1 --time=01:00:00 -p gpu_test --gres=gpu:4 --cpus-per-task=10 --mem-per-cpu=32G"
    pre_script = "cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} torchrun --nnodes 1 --nproc_per_node 4 --master_port 29501 llm-recipes/finetuning.py --project_name test --model_name {params['model_name']} --dataset.file {params['dataset']} --lr {params['lr']} --num_epochs {params['num_epochs']} --batch_size_training {params['batch_size_training']} --val_batch_size {params['val_batch_size']} --final_div_factor {params['final_div_factor']} --output_dir {output_path} --distillation_config.model_name {params['teacher_model_name']} --distillation --distillation_config.enable_fsdp --distillation_config.pure_bf16 --distillation_config.distil_factor {params['distil_factor']} --distillation_config.temperature {params['temperature']} --save_step 300 --seed 4\""
    subprocess.call(command ,shell=True)