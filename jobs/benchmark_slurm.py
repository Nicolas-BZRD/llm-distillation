import subprocess
from itertools import product

param_grid = {
    'model_id': ["meta-llama/Llama-2-7b-chat-hf"],
    'model_tokenizer': ["meta-llama/Llama-2-7b-chat-hf"],
    'dataset_id': ['/gpfs/users/boizardni/llm-distillation/datasets/processed/dialogsum'],
    'split_name': ['test'],
    'batch_size': [2],
    'number_few_shot': [4],
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    const = "--job-name=benchmark --nodes=1 --time=01:30:00 -p gpua100 --gres=gpu:1 --cpus-per-task=8 --mem-per-cpu=32G"
    pre_script = "cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} python llm-distillation/benchmark/benchmark.py --model_id {params['model_id']} --model_tokenizer {params['model_tokenizer']} --dataset_id {params['dataset_id']} --split_name {params['split_name']} --batch_size {params['batch_size']} --number_few_shot {params['number_few_shot']} --bfloat --context --task summary_dialogue --from_disk --mapping /gpfs/users/boizardni/llm-distillation/benchmark/mapping/dialogsum.json --bert_score\""
    subprocess.call(command ,shell=True)