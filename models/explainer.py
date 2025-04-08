import numpy as np
from IPython.display import display, HTML
import re
from lime.lime_text import LimeTextExplainer
from .predictor import LstmPredictor

class LimeExplainer:
    CLASS_NAMES = ['-1', '0', '1']
    INVISIBLE_A = "\u2060"
    INVISIBLE_B = "\u2061"
    NUM_SAMPLES = 4000 # empirically good in terms of convergence and speed
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

    def explain(self, dialogues, num_features=NUM_KEYWORDS):
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
        

if __name__ == '__main__':
    predictor = LstmPredictor()
    Lime = LimeExplainer(predictor)
    dialogues = [
        [
            {'role': 'A', 'response': 'Hey Alex! Saw your profile and felt a connection. Wanna grab dinner tonight? 😊'}, 
            {'role': 'B', 'response': "Absolutely! Let's meet at 7 PM? Looking forward to it! 😊"}, 
            {'role': 'A', 'response': "Great! Can't wait to know you better. 😊"}, 
            {'role': 'B', 'response': 'Guess we both like being direct! See you soon, Nur. 😊'}
        ],
    ]
    print(Lime.explain(dialogues))