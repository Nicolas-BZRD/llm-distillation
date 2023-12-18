import subprocess
from itertools import product

param_grid = {
    'model_id': ["/gpfs/users/boizardni/llm-distillation/train/models/distillation/pythia-410m-deduped_dist0.5_temp1_lr1e-06/9854"],
    # 'model_tokenizer': ["meta-llama/Llama-2-7b-hf"],
    'model_tokenizer': ["EleutherAI/pythia-410m-deduped"],
    'dataset_id': ['squad'],
    'split_name': ['validation'],
    'batch_size': [4],
    'number_few_shot': [0],
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    const = "--job-name=benchmark --nodes=1 --time=01:00:00 -p gpu_test --gres=gpu:1 --cpus-per-task=8 --mem-per-cpu=32G"
    pre_script = "cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} python llm-distillation/benchmark/benchmark.py --model_id {params['model_id']} --model_tokenizer {params['model_tokenizer']} --dataset_id {params['dataset_id']} --split_name {params['split_name']} --batch_size {params['batch_size']} --number_few_shot {params['number_few_shot']}\""
    subprocess.call(command ,shell=True)