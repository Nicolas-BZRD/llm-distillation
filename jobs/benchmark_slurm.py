import subprocess
from itertools import product

param_grid = {
    'model_id': ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf", "tiiuae/falcon-7b-instruct"],
    # 'model_tokenizer': ["EleutherAI/pythia-410m-deduped"],
    'dataset': [
        ['GEM/FairytaleQA', 'qa_generative'],
        ['squad', 'qa'],
        ['/gpfs/users/boizardni/llm-distillation/datasets/processed/dialogsum', 'summary_dialogue'],
        ['/gpfs/users/boizardni/llm-distillation/datasets/processed/pubmed_qa', 'qa_medical'],
        ['/gpfs/users/boizardni/llm-distillation/datasets/processed/qed', 'qa']
    ],
    'number_few_shot': [0, 1, 2, 3, 4, 5],
}
param_names = param_grid.keys()

for param_values in product(*param_grid.values()):
    params = dict(zip(param_names, param_values))

    const = "--job-name=benchmark --nodes=1 --time=03:00:00 -p gpua100 --gres=gpu:1 --cpus-per-task=8 --mem-per-cpu=32G"
    pre_script = "cd /gpfs/users/boizardni/; module load anaconda3/2020.02/gcc-9.2.0; source activate llm_distillation;"

    subprocess.call(f"mkdir /gpfs/users/boizardni/llm-distillation/benchmark/results/{params['model_id'].split('/')[-1]}" ,shell=True)
    subprocess.call(f"mkdir /gpfs/users/boizardni/llm-distillation/benchmark/results/{params['model_id'].split('/')[-1]}/{params['dataset'][0].split('/')[-1]}" ,shell=True)

    if params['dataset'][0] in ['squad', '/gpfs/users/boizardni/llm-distillation/datasets/processed/qed']: split_name = 'validation'
    else: split_name = 'test'

    command = f"sbatch {const} --wrap=\"{pre_script} python llm-distillation/benchmark/benchmark.py --model_id {params['model_id']} --dataset {params['dataset'][0]} --split_name {split_name} --batch_size 2 --number_few_shot {params['number_few_shot']} --task {params['dataset'][1]} --bfloat --bert_score --save_predictions"

    # Mapping column name
    if params['dataset'][0] in ['/gpfs/users/boizardni/llm-distillation/datasets/processed/dialogsum', 'GEM/FairytaleQA', '/gpfs/users/boizardni/llm-distillation/datasets/processed/qed']:
        mapping = f"/gpfs/users/boizardni/llm-distillation/benchmark/mapping/{params['dataset'][0].split('/')[-1]}.json"
        command += f" --mapping {mapping}"

    # Local dataset
    if params['dataset'][0] in ['/gpfs/users/boizardni/llm-distillation/datasets/processed/dialogsum', '/gpfs/users/boizardni/llm-distillation/datasets/processed/pubmed_qa', '/gpfs/users/boizardni/llm-distillation/datasets/processed/qed']:
        command += " --from_disk"

    # Mapping dict
    if params['dataset'][0] in ['/gpfs/users/boizardni/llm-distillation/datasets/processed/qed']:
        command += " --mapping_dict string"

    # Un-titled
    if params['dataset'][0] in ['squad', 'GEM/FairytaleQA', '/gpfs/users/boizardni/llm-distillation/datasets/processed/qed']:
        subprocess.call(f"mkdir /gpfs/users/boizardni/llm-distillation/benchmark/results/{params['model_id'].split('/')[-1]}/{params['dataset'][0].split('/')[-1]}/titled" ,shell=True)
        subprocess.call(f"mkdir /gpfs/users/boizardni/llm-distillation/benchmark/results/{params['model_id'].split('/')[-1]}/{params['dataset'][0].split('/')[-1]}/untitled" ,shell=True)

        subprocess.call(command + "\"",shell=True)
        subprocess.call(command + " --title\"",shell=True)
    else:
        subprocess.call(f"mkdir /gpfs/users/boizardni/llm-distillation/benchmark/results/{params['model_id'].split('/')[-1]}/{params['dataset'][0].split('/')[-1]}/untitled" ,shell=True)
        subprocess.call(command+"\"",shell=True)