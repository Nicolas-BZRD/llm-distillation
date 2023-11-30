import subprocess
from itertools import product

param_grid = {
    'model_id': ['meta-llama/Llama-2-7b-hf',],
    'model_tokenizer': ["meta-llama/Llama-2-7b-hf"],
    'dataset_id': ['squad'],
    'split_name': ['train'],
    'number_few_shot': [2],
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    const = "--job-name=generator --nodes=1 --time=04:00:00 -p gpua100 --gres=gpu:1 --cpus-per-task=8 --mem-per-cpu=32G"
    pre_script = "cd /gpfs/users/boizardni/llm-distillation; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} python datasets/generator.py --model_id {params['model_id']} --model_tokenizer {params['model_tokenizer']} --dataset_id {params['dataset_id']} --split_name {params['split_name']} --number_few_shot {params['number_few_shot']} --context --bfloat\""
    subprocess.call(command ,shell=True)