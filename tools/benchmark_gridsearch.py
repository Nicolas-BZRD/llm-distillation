import subprocess
from itertools import product

param_grid = {
    'model_id': ['meta-llama/Llama-2-7b-hf', 'EleutherAI/pythia-410m-deduped'],
    'dataset_id': ['squad', 'squad_v2'],
    'number_few_shot': [0, 2, 5],
    'context': [False, True]
}

for params in list(product(*param_grid.values())):
    model_id = params[0]
    dataset_id = params[1]
    number_few_shot = params[2]
    context_bool = params[3]

    context_id = -1
    if context_bool:
        if dataset_id == 'squad':
            context_id = 0
        elif dataset_id == "squad_v2":
            context_id = 1

    command = f"sbatch --nodes=1 --time=1:00:00 -p gpua100 --gres=gpu:1 --cpus-per-task=4  --wrap=\"module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation; cd llm_distillation/benchmark/; python benchmark.py --model_id {model_id} --dataset_id {dataset_id} --context_id {context_id} --number_few_shot {number_few_shot}\""
    subprocess.call(command ,shell=True)