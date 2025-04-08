import numpy as np
from IPython.display import display, HTML
import re
from lime.lime_text import LimeTextExplainer
from .predictor import LstmPredictor

class LimeExplainer:
    CLASS_NAMES = ['-1', '0', '1']
    INVISIBLE_A = "\u2060"
    INVISIBLE_B = "\u2061"
    NUM_SAMPLES = 200 # empirically good in terms of convergence and speed
    NUM_KEYWORDS = 10 # how many keywords to highlight in one dialogue
    
    def __init__(self, predictor: LstmPredictor):
        self.predictor = predictor

    def convert_text_to_dialogue(self, text):
        dialogue = []
        segments = re.split(f"({self.INVISIBLE_A}|{self.INVISIBLE_B})", text)

        role_map = {self.INVISIBLE_A: 'A', self.INVISIBLE_B: 'B'}
        current_role = None
        for seg in segments:
            if seg in role_map:
                current_role = role_map[seg]
            elif current_role and seg.strip():
                dialogue.append({'role': current_role, 'response': seg.strip()})
        if not dialogue:
            dialogue = [{'role': 'A', 'response': text.strip()}]
        return dialogue
    
    def predict_proba_wrapper(self, text_samples):
        results = []
        for text in text_samples:
            dialogue = self.convert_text_to_dialogue(text)
            # Here, predictor.predict expects a list of dialogues, so wrap dialogue in a list
            prob = predictor.predict([dialogue], explainer_mode=True)
            # Assuming predictor.predict returns a list with one element for each input dialogue
            results.append(prob[0])
        probs = np.array(results)
        return probs

    def explain(self, dialogues, num_features=NUM_KEYWORDS, sentence_level=False):
        if not sentence_level:
            text = ""
            for sentence in dialogues[0]:
                marker = self.INVISIBLE_A if sentence['role'] == 'A' else self.INVISIBLE_B
                text += marker + sentence['response'] + " "
            explainer = LimeTextExplainer(class_names=self.CLASS_NAMES)
            explanation = explainer.explain_instance(
                text,
                self.predict_proba_wrapper,
                num_features=num_features,
                num_samples=self.NUM_SAMPLES
            )
            return explanation.as_list()
        else:
            dialogue_xs = dialogues[0]
            length = len(dialogue_xs)
            if length == 0:
                return []
            i = 0
            curr_dialogue = [dialogue_xs[i]]
            log = ''
            while i < length:
                curr_dialogue.append(dialogue_xs[i])
                label = self.predictor.predict([curr_dialogue])
                log += f"Until sentence {i}, the label is {label}\n"
                i += 1
            return log



if __name__ == '__main__':
    predictor = LstmPredictor()
    Lime = LimeExplainer(predictor)
    dialogues = [
        [
            {"role": "A", "response": "$S$"},
            {"role": "B", "response": "Hey, Cheryl! Are you free tonight for dinner?"},
            {"role": "A", "response": "Hey Jinwoo, that sounds nice. Same place as last time or do you have somewhere new in mind?"},
            {"role": "B", "response": "Let's try something new! How about the cozy café by the Marina Bay?"},
            {"role": "A", "response": "Sounds perfect. I've been wanting to visit Marina Bay again."},
            {"role": "B", "response": "Great, I'll meet you there at 7 PM. Looking forward to it!"},
            {"role": "A", "response": "Can't wait to catch up and explore Marina Bay!"},
            {"role": "B", "response": "Really excited to see you tonight, Cheryl."},
            {"role": "A", "response": "Me too! Can't wait to hear about your recent adventures."},
            {"role": "B", "response": "Can't wait to share them over a nice meal!"},
            {"role": "A", "response": "I'll make sure to bring my favorite playlist for a perfect evening."},
            {"role": "B", "response": "Awesome! See you soon."},
            {"role": "A", "response": "$S$"},
            {"role": "B", "response": "Can't wait, can't help being straightforward—this means a lot!"},
            {"role": "A", "response": "That means a lot to me too, Jinwoo. See you soon!"},
            {"role": "B", "response": "See you soon, Cheryl! Ready for an amazing night."},
            {"role": "A", "response": "Absolutely, let's make it special!"},
            {"role": "B", "response": "Let's make tonight unforgettable, Cheryl!"},
            {"role": "A", "response": "Excited to create new memories with you tonight!"},
            {"role": "B", "response": "I'm really grateful to have this opportunity with you."}
        ]
    ]
    print(Lime.explain(dialogues, sentence_level=False))