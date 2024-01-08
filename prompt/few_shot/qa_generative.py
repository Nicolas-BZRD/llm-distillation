data = [
  {
    "title": "the-wee-bannock",
    "context": "So, she jumped up with her lint and her lint cards, and the tailor jumped up with his great shears, and one apprentice grasped the line measure, while another took up the saucer full of pins; and they all tried to catch the wee bannock. But it dodged them round and round the fire, and at last it got safely out of the door and ran down the road, with one of the apprentices after it, who tried to snip it in two with his shears. It ran too quickly for him, however, and at last he stopped and went back to the house, while the wee bannock ran on until it came to a tiny cottage by the roadside. it trundled in at the door, and there was a weaver sitting at his loom, with his wife beside him, winding a clue of yarn.",
    "question": "How did the bannock escape from the tailor's wife and the three tailors?",
    "answers": "Dodged them round and round the fire, and at last it got safely out of the door and ran down the road.",
    "sources": "FairytaleQA"
  },
  {
    "title": "princess-glass-mountain",
    "context": "Then he took the prince by the hand, led him deep down in the earth into his cave, and there on the wall hung a suit of armor altogether forged of the clearest silver, and so bright that it shone afar. Right beside it stood a snow - white steed, saddled and bridled, pawing the earth with his silver hoofs, and champing his bit till the foam dropped to the ground. The wild man said: 'now get quickly into your armor, ride out and try your luck! in the meantime I will tend your oxen.' The prince did not wait to be told a second time; but put on his helmet and armor in all haste, securely buckled on his spurs, hung his sword at his side, and felt as light in his silver armor as a bird in the air. Then he leaped into the saddle so that every clasp and buckle rang, laid his reins on the neck of his steed, and rode hastily toward the glass mountain.",
    "question": "What was the suit of armor given by the wild man forged from?",
    "answers": "The clearest silver.",
    "sources": "FairytaleQA"
  },
  {
    "title": "money-box",
    "context": "He knew very well that he had enough inside him to buy up all the other toys, and this gave him a very good opinion of his own value. The rest thought of this fact also, although they did not express it, for there were so many other things to talk about. A large doll, still handsome, though rather old, for her neck had been mended, lay inside one of the drawers which was partly open. She called out to the others, 'let us have a game at being men and women, that is something worth playing at.'",
    "question": "Why didn't the other toys talk about how valuable the pig was?",
    "answers": "There were so many other things to talk about.",
    "sources": "FairytaleQA"
  },
  {
    "title": "a-legend-of-confucius",
    "context": "When confucius came to the earth, the kilin, that strange beast which is the prince of all four - footed animals, and only appears when there is a great man on earth, sought the child and spat out a jade whereon was written: 'son of the watercrystal you are destined to become an uncrowned king!' and confucius grew up, studied diligently, learned wisdom and came to be a saint. He did much good on earth, and ever since his death has been reverenced as the greatest of teachers and masters. He had foreknowledge of many things and even after he had died, he gave evidence of this.",
    "question": "Why was confucius's death reverenced as the greatest of teachers and masters?",
    "answers": "He did much good on earth.",
    "sources": "FairytaleQA"
  },
  {
    "title": "naughty-boy",
    "context": "'Oh, let me in! Let me in! I'm cold, and I'm so wet!' Exclaimed suddenly a child that stood crying at the door and knocking for admittance, while the rain poured down, and the wind made all the windows rattle. 'Poor thing!' said the old poet, as he went to open the door. there stood a little boy, quite naked, and the water ran down from his long golden hair. He trembled with cold, and had he not come into a warm room he would most certainly have perished in the frightful tempest.",
    "question": "Why did the boy ask to come inside?",
    "answers": "He was cold and wet.",
    "sources": "FairytaleQA"
  }
]

def create_few_shot(number_few_shot: int, **args):
  shot = []
  for i in range(number_few_shot):
    if args.get('title', False):
      shot.append(
        [
          f"Title: {data[i]['title']}\nContext: {data[i]['context']}\nQuestion: {data[i]['question']}",
          f"Answer: {data[i]['answers']}"
        ]
      )
    else:
      shot.append(
        [
          f"Context: {data[i]['context']}\nQuestion: {data[i]['question']}",
          f"Answer: {data[i]['answers']}"
        ]
      )
  return shot

def create_request(title="", context="", question="", **args):
  if title: return [f"Title: {title}\nContext: {context}\nQuestion: {question}", "Answer:"]
  else: return [f"Context: {context}\nQuestion: {question}", "Answer:"]