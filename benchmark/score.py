import re
import string
import secrets
import evaluate
from collections import Counter

def _normalize(s):
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def _f1_score_sentence(prediction, answer):
    prediction_tokens = prediction.split()
    answer_tokens = answer.split()
    
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0, 0, 0
    
    precision = num_common / len(prediction_tokens)
    recall = num_common / len(answer_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall

def f1_score(predictions, answers):
    f1_scores, precision_scores, recall_scores = [], [], []

    for prediction, answer_list in zip(predictions, answers):
        prediction = _normalize(prediction)
        max_f1, max_precision, max_recall = 0, 0, 0

        if not answer_list:
          if prediction == "" or 'no response' in prediction:
            max_f1 = max_precision = max_recall = 1
          else:
            max_f1 = max_precision = max_recall = 0
        else:
          if isinstance(answer_list, str): answer_list = [answer_list]
          for answer in answer_list:
            answer = _normalize(answer)
            f1, precision, recall = _f1_score_sentence(prediction, answer)
            max_f1, max_precision, max_recall = max(f1, max_f1), max(precision, max_precision), max(recall, max_recall)

        f1_scores.append(max_f1)
        precision_scores.append(max_precision)
        recall_scores.append(max_recall)

    average_f1 = sum(f1_scores) / len(f1_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)

    return {'f1': average_f1, 'precision': average_precision, 'recall': average_recall}

def exact_match(predictions, answers):
    exact_match_scores = []
    for prediction, answer_list in zip(predictions, answers):
        prediction = _normalize(prediction)
        if isinstance(answer_list, str): answer_list = [answer_list]
        answer_list = [_normalize(item) for item in answer_list]
        if not answer_list and prediction == "" or "no response" in prediction: exact_match_scores.append(1)
        if prediction in answer_list: exact_match_scores.append(1)
        else: exact_match_scores.append(0)
    return sum(exact_match_scores)/len(exact_match_scores)

def rouge(predictions, answers):
    rouge_metric = evaluate.load('rouge', experiment_id=f"{secrets.randbelow(10000)}")
    return rouge_metric.compute(predictions=predictions, references=answers)

def bert_score(predictions, answers):
    bertscore = evaluate.load("bertscore", experiment_id=f"{secrets.randbelow(10000)}")
    if isinstance(answers[0], dict):
      f1, precision, recall = [0]*len(predictions), [0]*len(predictions), [0]*len(predictions)
      for i, row in enumerate(answers):
        for answer in row:
          tmp = bertscore.compute(predictions=predictions[i], references=answer, lang="en", rescale_with_baseline=True)
          f1[i] = max(f1[i], tmp['f1'])
          precision[i] = max(precision[i], tmp['precision'])
          recall[i] = max(recall[i], tmp['recall'])
      return {'f1': f1, 'precision': precision, 'recall': recall}
    else:
      return bertscore.compute(predictions=predictions, references=answers, lang="en", rescale_with_baseline=True)