import os
import json

def __create_few_shot(number_few_shot, title=True):
    with open(f"{os.getenv('HOME')}/llm-distillation/tools/qa/few_shot_examples.json") as json_file:
        data = json.load(json_file)

    templates = [
        "Title: {title}\nPassage: {context}\nQuestion: {question}\nAnswer: {answers}",
        "Passage: {context}\nQuestion: {question}\nAnswer: {answers}",
    ]

    if title:
        prompt = "\n\n".join([
            templates[0].format(title=row['title'], context=row['context'], question=row['question'], answers=row['answers']) for row in data[0:number_few_shot]
        ]) + '\n\n'
    else:
        prompt = "\n\n".join([
            templates[1].format(context=row['context'], question=row['question'], answers=row['answers']) for row in data[0:number_few_shot]
        ]) + '\n\n'
    
    return prompt


def create_pre_prompt(context=False, title=False, few_shot=0):
    pre_prompt = ""
    if context:
        with open(f"{os.getenv('HOME')}/llm-distillation/tools/context.json") as json_file:
            pre_prompt = json.load(json_file)['qa']

    if few_shot:
        pre_prompt += __create_few_shot(number_few_shot=few_shot, title=title)
    return pre_prompt


def create_prompt(pre_prompt="", title="", context="", question=""):
    if title:
        template = "Title: {title}\nPassage: {context}\nQuestion: {question}\nAnswer:"
        prompt = template.format(title=title, context=context, question=question)
    else:
        template = "Passage: {context}\nQuestion: {question}\nAnswer:"
        prompt = template.format(title=title, context=context, question=question)
        
    return pre_prompt + prompt