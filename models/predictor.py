from typing import Literal
import numpy as np
import torch
from pydantic import BaseModel
import logging
import spacy
from IPython.display import display, HTML
import torch.nn.functional as F

from .lstm_basic import LstmBasic
from .lstm_advanced import LstmAdvanced
from lime.lime_text import LimeTextExplainer


class TextData(BaseModel):
    role: Literal["A", "B"]
    response: str

class LstmPredictor:
    MODEL_PATHS = set([
        "glove.6B.50d", "glove.6B.100d", "glove.6B.200d",
        "glove.6B.300d", "glove.42B.300d", "glove.840B.300d",
    ])
    MICRO_SEQUENCE_LENGTH = 50
    MACRO_SEQUENCE_LENGTH = 25

    def __init__(self,
                 variant: str = "glove.42B.300d",
                 with_pause: bool = True,
                 advanced: bool = False
                 ):
        """
        Instantiates a new instance for prediction.
        The default parameter values are those
        I believe will produce the best results.

        This constructor loads all necessary data for evaluation:
        glove embeddings, spacy language, neural network itself.
        This can take up quite a lot of time.
        Please reuse instance where possible.
        
        Parameters
        ---
        variant: str
            The model variant to load. Must be one of
            `"glove.6B.50d"`, `"glove.6B.100d"`, `"glove.6B.200d"`,
            `"glove.6B.300d"`, `"glove.42B.300d"`, "glove.840B.300d"`
        with_pause: bool
            Whether to consider the pauses (in the dialogues) in prediction
        advanced: bool
            When `advanced=True`, use advanced model architecture
            as defined in `models/lstm_advanced.py`.
            When `advanced=False`, use basic model architecture
            as defined in `models/lstm_basic.py`.
        """
        self.logger = logging.getLogger("LstmPredictor")
        if variant not in LstmPredictor.MODEL_PATHS:
            raise ValueError(
                f"Invalid model: \"{variant}\", should be one of {LstmPredictor.MODEL_PATHS}")
        self.vector_size = int(variant.split(".")[-1][:-1])
        if advanced:
            self.model = LstmAdvanced(self.vector_size)
            prefix = "lstm_advanced_0" if with_pause else "lstm_advanced"
        else:
            self.model = LstmBasic(self.vector_size)
            prefix = "lstm_basic_0" if with_pause else "lstm_basic"
        self.device = self._find_device()
        self.model = torch.nn.DataParallel(self.model).to(self.device)
        self.model.load_state_dict(torch.load(f"models/{prefix}.{variant}.pth", weights_only=True))
        self.embeddings = self._load_embeddings(f"vectors/{variant}.txt")
        self.nlp = spacy.load("en_core_web_sm")
    
    def _load_embeddings(self, path: str):
        embeddings = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split(" ")
                word = values[0]
                arr = [float(item) if item != "." else 0.0 for item in values[1:]]
                vector = np.array(arr)
                embeddings[word] = vector
        self.logger.info(f"Embeddings loaded with size: {len(embeddings)}")
        return embeddings
    
    def _find_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {device}")
        return device

    def predict(self, dialogues: list[list], explainer_mode=False):
        """
        Predicts the label of the given list of dialogues.

        Parameters
        ---
        dialogues: list[list]
            The list of dialogues.
            Each element is a dialogue, which is a list of text,
            each text is a dictionary that has keys `"role"` and `"response"`,
            representing a text message.
        
        Returns
        ---
        If in explainer mode, returns probabilities of each class.
        Else, return list[Literal[-1, 0, 1]] (The list of corresponding labels)
        """
        X = self._preprocess(dialogues)
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            probas = F.softmax(logits, dim=1).cpu().detach().numpy()
        if explainer_mode:
            # For explainer mode, return the probabilities
            return probas
        else:
            preds = np.argmax(probas, axis=1) - 1
            return list(preds)
    
    def _preprocess(self, dialogues: list[list]) -> torch.Tensor:
        processed_dialogues = []
        for dialogue in dialogues:
            dialogue_data = []
            for text in dialogue:
                if not isinstance(text, dict):
                    raise ValueError(
                        f"Each text must be a dictionary with keys \"role\" and \"response\", "
                         "but received type {type(text)}")
                text_data = TextData(**text)
                doc = self.nlp(text_data.response)
                embeddings_data = []
                for token in doc:
                    if token.lemma_ in self.embeddings:
                        embeddings_data.append(self.embeddings[token.lemma_])
                l_0 = len(embeddings_data)
                # This may cause issues because we are ignoring sentences only with emojis
                if l_0 == 0:
                    continue
                assert l_0 <= LstmPredictor.MICRO_SEQUENCE_LENGTH, \
                    "Found text with longer sequence length than those this model is trained on: " \
                    f"\"{text_data.response}\". " \
                    "You may need to increase the length limit of micro sequence, " \
                    "but please discuss with Nguyen first."
                embeddings = np.array(embeddings_data)
                embeddings = np.vstack((
                    np.zeros((LstmPredictor.MICRO_SEQUENCE_LENGTH - l_0, self.vector_size)),
                    embeddings))
                dialogue_data.append(embeddings)
            L_0 = len(dialogue_data)
            assert L_0 <= LstmPredictor.MACRO_SEQUENCE_LENGTH, \
                "Found a dialogue longer than those this model is trained on: " \
                f"{dialogue} " \
                "You may need to increase the length limit of macro sequence, " \
                "but please discuss with Nguyen first."
            dialogue_data = np.array(dialogue_data)
            dialogue_data = np.vstack((
                np.zeros((LstmPredictor.MACRO_SEQUENCE_LENGTH - L_0, LstmPredictor.MICRO_SEQUENCE_LENGTH, self.vector_size)),
                dialogue_data))
            processed_dialogues.append(dialogue_data)
        return torch.tensor(np.array(processed_dialogues), dtype=torch.float32)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # No. 1
    predictor = LstmPredictor()
    dialogues = [
        [{"role": "A", "response": "They are so beautiful aren’t they"}, {"role": "B", "response": "Good morning sunshine\nThe next time I’m distracting you from learning please say that"}, {"role": "A", "response": "Morning honey"}, {"role": "A", "response": "I will\nBut as I have never said that I think you should have known what I mean\nAnd today finally I can talk with my supervisor about my future plans. She is sooooo cute"}, {"role": "B", "response": "😢I will steal every cute girl from you"}],
    ]
    labels = predictor.predict(dialogues, explainer_mode=True) # set explainer_mode=True to get probabilities
    print(labels)
