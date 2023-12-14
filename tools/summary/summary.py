import os
import json

def __create_few_shot(number_few_shot):
    with open(f"{os.getenv('HOME')}/llm-distillation/tools/summary/few_shot_examples.json") as json_file:
        data = json.load(json_file)

    templates = "Text: {text}\nSummary: {summary}"

    prompt = "\n\n".join([
        templates.format(text=row['text'], summary=row['summary']) for row in data[0:number_few_shot]
    ]) + '\n\n'

    return prompt


def create_pre_prompt(context=False, few_shot=0):
    pre_prompt = ""
    if context:
        with open(f"{os.getenv('HOME')}/llm-distillation/tools/context.json") as json_file:
            pre_prompt = json.load(json_file)['summary']

    if few_shot:
        pre_prompt += __create_few_shot(number_few_shot=few_shot)
    return pre_prompt


def create_prompt(pre_prompt="", text=""):
    template = "Text: {text}\nSummary:"
    prompt = template.format(text=text)
    return pre_prompt + prompt