import os
import json
import importlib
from pathlib import Path

def __load_module_from_py_file(py_file: str) -> object:
    module_name = Path(py_file).name
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    spec = importlib.util.spec_from_loader(module_name, loader)
    module = importlib.util.module_from_spec(spec)

    loader.exec_module(module)
    return module

def create_prompt(task: str, few_shot: int, *args):
    prompt = ""
    if args[0].get('sys_prompt', False):
        with open(f"{os.getenv('HOME')}/llm-distillation/prompt/context.json") as json_file:
            prompt += json.load(json_file)[task]+"\n\n"

    module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
    request = getattr(module, "create_request")(**args[0])

    if few_shot:
        module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
        shot = getattr(module, "create_few_shot")(few_shot, args[0])
        prompt += ('\n\n').join(shot) + f"\n\n{request}"
    else:
        prompt += request
    return prompt   

def llama_chat_prompt(task: str, few_shot: int, *args):
    prompt = ""
    with open(f"{os.getenv('HOME')}/llm-distillation/prompt/context.json") as json_file:
        prompt += f"<s>[INST] <<SYS>>\n{json.load(json_file)[task]}\n<</SYS>>\n\n"

    module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
    request = getattr(module, "create_request")(**args[0])

    if few_shot:
        module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
        shot = getattr(module, "create_few_shot")(few_shot, args[0])
        prompt += ('\n\n').join(shot) + f"\n\n{request}"
    else:
        prompt += request
    return prompt        

# Old version
# def llama_chat_prompt(task: str, few_shot: int, *args):
#     prompt = ""
#     with open(f"{os.getenv('HOME')}/llm-distillation/prompt/context.json") as json_file:
#         prompt += f"<s>[INST] <<SYS>>\n{json.load(json_file)[task]}\n<</SYS>>\n\n"

#     module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
#     request = getattr(module, "create_request")(**args[0])

#     if few_shot:
#         module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
#         shot = getattr(module, "create_few_shot")(few_shot, args[0])
#         for i in range(len(shot)):
#             if i == 0:
#                 prompt += f"{shot[i][0]} [/INST] {shot[i][1]} <\s>"
#             else:
#                 prompt += f"<s>[INST] {shot[i][0]} [/INST] {shot[i][1]} <\s>"
#         prompt += f"<s>[INST] {request[0]} [/INST] {request[1]}"
#     else:
#         prompt += f"{request[0]} [/INST] {request[1]}"

#     return prompt