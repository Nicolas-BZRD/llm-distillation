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

def create_prompt(task: str, few_shot: int, sys_prompt: bool, *args):
    prompt = ""
    if sys_prompt:
        with open(f"{os.getenv('HOME')}/llm-distillation/prompt/context.json") as json_file:
            prompt += json.load(json_file)[task] + "\n\n"

    module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
    request = getattr(module, "create_request")(**args[0])

    if few_shot:
        shot = getattr(module, "create_few_shot")(few_shot, args[0])
        shot = ['\n'.join(s) for s in shot]
        request = '\n'.join(request)
        prompt += '\n\n'.join(shot) + f"\n\n{request}"
    else:
        request = '\n'.join(request)
        prompt += request
    return prompt   

def create_chat_prompt(task: str, few_shot: int, chat_template, sys_in_chat=False, *args):
    chat=[]
    with open(f"{os.getenv('HOME')}/llm-distillation/prompt/context.json") as json_file:
        sys_prompt = json.load(json_file)[task]

    if not sys_in_chat: chat.append({"role": "system", "content": sys_prompt})

    module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
    request = getattr(module, "create_request")(**args[0])

    if few_shot:
        module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
        shot = getattr(module, "create_few_shot")(few_shot, args[0])
        if sys_in_chat:
            chat.append({"role": "user", "content": f"{sys_prompt}\n\n{shot[0][0]}"})
            chat.append({"role": "assistant", "content": shot[0][1]})
            shot = shot[1:]
        for s in shot:
            chat.append({"role": "user", "content": s[0]})
            chat.append({"role": "assistant", "content": s[1]})
        chat.append({"role": "user", "content": request[0]})
        chat.append({"role": "assistant", "content": request[1]})
    else:
        if sys_in_chat:
            chat.append({"role": "user", "content": sys_prompt + '\n\n' + request[0]})
            chat.append({"role": "assistant", "content": shot[0][1]})
            shot = shot[1:]
        chat.append({"role": "user", "content": request[0]})
        chat.append({"role": "assistant", "content": request[1]})
    prompt = chat_template(chat, tokenize=False)
    return prompt[:prompt.rfind("Answer:")+7]

if __name__ == "__main__":
    from transformers import AutoTokenizer

    print(
        create_prompt("qa", 0, True,
            {
                "title": "mammography-quality-standards-act",
                "context": "The United States Food and Drug Administration implemented federal regulations governing mammography under the Mammography Quality Standards Act (MQSA) of 1992. During 1995, its first year in implementation, we examined the impact of the MQSA on the quality of mammography in North Carolina. All mammography facilities were inspected during 1993-1994, and again in 1995. Both inspections evaluated mean glandular radiation dose, phantom image evaluation, darkroom fog, and developer temperature. Two mammography health specialists employed by the North Carolina Division of Radiation Protection performed all inspections and collected and codified data. The percentage of facilities that met quality standards increased from the first inspection to the second inspection. Phantom scores passing rate was 31.6% versus 78.2%; darkroom fog passing rate was 74.3% versus 88.5%; and temperature difference passing rate was 62.4% versus 86.9%.",
                "question": "Has the mammography quality standards act affected the mammography quality in North Carolina?"
            }
        )
    )

    print("--------------------------------")

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    print(
        create_chat_prompt("qa", 3, tokenizer.apply_chat_template, True,
            {
                "title": "mammography-quality-standards-act",
                "context": "The United States Food and Drug Administration implemented federal regulations governing mammography under the Mammography Quality Standards Act (MQSA) of 1992. During 1995, its first year in implementation, we examined the impact of the MQSA on the quality of mammography in North Carolina. All mammography facilities were inspected during 1993-1994, and again in 1995. Both inspections evaluated mean glandular radiation dose, phantom image evaluation, darkroom fog, and developer temperature. Two mammography health specialists employed by the North Carolina Division of Radiation Protection performed all inspections and collected and codified data. The percentage of facilities that met quality standards increased from the first inspection to the second inspection. Phantom scores passing rate was 31.6% versus 78.2%; darkroom fog passing rate was 74.3% versus 88.5%; and temperature difference passing rate was 62.4% versus 86.9%.",
                "question": "Has the mammography quality standards act affected the mammography quality in North Carolina?"
            }
        )
    )