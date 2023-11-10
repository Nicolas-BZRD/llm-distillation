import subprocess
from itertools import product

param_grid = {
    'model_id': [
        '/gpfs/users/boizardni/llm-distillation/train/models/finetuned/pythia-410m-deduped/squad',
        '/gpfs/users/boizardni/llm-distillation/train/models/finetuned/pythia-410m-deduped/Llama-2-7b-hf_squad_train_2s_0context'    
    ],
    'model_tokenizer': ['EleutherAI/pythia-410m-deduped']
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    const = "--job-name=benchmark --nodes=1 --time=02:00:00 -p gpua100 --gres=gpu:1 --cpus-per-task=4"
    pre_script = "cd /gpfs/users/boizardni/llm-distillation/benchmark/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"
    command = f"sbatch {const} --wrap=\"{pre_script} python benchmark.py --model_id {params['model_id']} --model_tokenizer {params['model_tokenizer']}\""
    subprocess.call(command ,shell=True)
