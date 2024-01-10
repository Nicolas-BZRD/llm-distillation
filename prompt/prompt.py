import os
import json
from importlib import machinery, util
from pathlib import Path


def __load_module_from_py_file(py_file: str) -> object:
    module_name = Path(py_file).name
    loader = machinery.SourceFileLoader(module_name, py_file)
    spec = util.spec_from_loader(module_name, loader)
    module = util.module_from_spec(spec)

    loader.exec_module(module)
    return module


def create_prompt(task: str, few_shot: int, **args):
    prompt = ""
    if args.get('sys_user', False):
        prompt += json.load(open(f"{os.getenv('HOME')}/llm-distillation/prompt/context.json"))[task] + "\n\n"

    module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
    request = '\n'.join(getattr(module, "create_request")(**args))

    if few_shot:
        shot = '\n\n'.join(['\n'.join(s) for s in getattr(module, "create_few_shot")(few_shot, **args)])
        prompt += f"{shot}\n\n{request}"
    else:
        prompt += request
    return prompt


def create_chat_prompt(task: str, few_shot: int, **args):
    chat, sys_prompt = [], json.load(open(f"{os.getenv('HOME')}/llm-distillation/prompt/context.json"))[task]
    module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
    request = getattr(module, "create_request")(**args)

    if not args.get('sys_user', False): chat.append({"role": "system", "content": sys_prompt})
    
    if few_shot:
        shot = getattr(module, "create_few_shot")(few_shot, **args)
        if args.get('sys_user', False):
            chat.extend([{"role": "user", "content": f"{sys_prompt}\n\n{shot[0][0]}"}, {"role": "assistant", "content": shot[0][1]}])
            shot = shot[1:]
        for s in shot: chat.extend([{"role": "user", "content": s[0]}, {"role": "assistant", "content": s[1]}])
        chat.extend([{"role": "user", "content": request[0]}, {"role": "assistant", "content": request[1]}])
    else:
        if args.get('sys_user', False): chat.extend([{"role": "user", "content": f"{sys_prompt}\n\n{request[0]}"}, {"role": "assistant", "content": request[1]}])
        else: chat.extend([{"role": "user", "content": request[0]}, {"role": "assistant", "content": request[1]}])
    prompt = args['chat_template'](chat, tokenize=False)

    if "qa" in task: return prompt[:prompt.rfind("Answer:") + 7]
    elif "summary" in task: return prompt[:prompt.rfind("Summary:") + 8]