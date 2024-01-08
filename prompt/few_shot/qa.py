data = [
  {
    "id": 72736,
    "title": "Christine's boyfriend",
    "context": "Patrick Harris (Tim DeKay), Old Christine's new boyfriend, who she meets in a video store and starts dating.",
    "question": "Who played patrick on new adventures of old christine?",
    "answers": "Tim DeKay",
    "sources": "Lexi/spanextract"
  },
  {
    "id": 0,
    "title": "June 14, 2018: Death Row Inmates",
    "context": "As of June 14, 2018, there were 2,718 death row inmates in the United States.",
    "question": "Total number of death row inmates in the us?",
    "answers": "2,718",
    "sources": "Lexi/spanextract"

  },
  {
      "id": 419,
      "title": "Modern Communism",
      "context": "Most modern forms of communism are grounded at least nominally in Marxism, an ideology conceived by noted sociologist Karl Marx during the mid nineteenth century.",
      "question": "Who came up with the idea of communism ?",
      "answers": "Karl Marx",
      "sources": "Lexi/spanextract"
  },
  {
    "id": 225,
    "title": "Napoleon's Defeat by Seventh Coalition",
    "context": "A French army under the command of Napoleon Bonaparte was defeated by two of the armies of the Seventh Coalition : a British-led Allied army under the command of the Duke of Wellington, and a Prussian army under the command of Gebhard Leberecht von Blücher, Prince of Wahlstatt.",
    "question": "Who commanded british forces at the battle of waterloo?",
    "answers": "The Duke of Wellington",
    "sources": "Lexi/spanextract"
  },
  {
  "id": 620,
  "title": "Canine character",
  "context": "Astro is a canine character on the Hanna-Barbera cartoon, The Jetsons.",
  "question": "What was the dog's name on the jetsons?",
  "answers": "Astro",
  "sources": "Lexi/spanextract"
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