from typing import Literal
import numpy as np
import torch
from pydantic import BaseModel
import logging
import spacy
from IPython.display import display, HTML

from .lstm_basic import LstmBasic
from .lstm_advanced import LstmAdvanced
from lime.lime_text import LimeTextExplainer
from .predictor import LstmPredictor, TextData

class LimeExplainer:
    CLASS_NAMES = ['-1', '0', '1']
    
    def __init__(self, predictor: LstmPredictor):
        self.predictor = predictor

    def convert_text_to_dialogue(self, text):
        return [{'role': 'A', 'response': text}]
    
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

if __name__ == '__main__':
    predictor = LstmPredictor()
    Lime = LimeExplainer(predictor)
    explainer = LimeTextExplainer(class_names=Lime.CLASS_NAMES)
    text = "Hi Priya! How's your day going? Want to grab dinner tonight?"
    explanation = explainer.explain_instance(text, Lime.predict_proba_wrapper, num_features=3)
    print(explanation.as_list())