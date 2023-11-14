import subprocess
from itertools import product

param_grid = {
    'model_id': [
        '/gpfs/users/boizardni/llm-distillation/train/models/finetuned/Llama-2-7b-hf/squad_2e-05/fine-tuned-meta-llama/Llama-2-7b-hf',
    ],
    'model_tokenizer': ['meta-llama/Llama-2-7b-hf']
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    const = "--job-name=benchmark --nodes=1 --time=02:00:00 -p gpua100 --gres=gpu:1 --cpus-per-task=8 --mem-per-cpu=32G"
    pre_script = "cd /gpfs/users/boizardni/llm-distillation/benchmark; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} python benchmark.py --model_id {params['model_id']} --model_tokenizer {params['model_tokenizer']} --batch_size 8 --bfloat\""
    subprocess.call(command ,shell=True)
