import ast
import pandas as pd
import numpy as np
import spacy
from typing import Literal
import pickle

Role = Literal[-1, 1]
Text = tuple[Role, np.ndarray]  # array shape: (L, V) where L is number of tokens, V is vector size (currently 50)
Dialog = list[Text]
Label = Literal[-1, 0, 1]

def get_embeddings(file: str) -> dict:
    embeddings = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector
    return embeddings

def preprocess_data(file: str = "Data/train.csv", embeddings_file: str = "vectors/glove.6B.50d.txt") -> list[tuple[Dialog, Label]]:
    embeddings = get_embeddings(embeddings_file)
    print("Embeddings loaded with size:", len(embeddings))
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(file)
    data: list[tuple[Dialog, Label]] = []
    for row in df.iloc:
        dialogue = ast.literal_eval(row['Dialogue'])['text']
        label = row['Label']
        dialogue_data: Dialog = []
        for text in dialogue:
            if text['response'] == '$S$':
                continue
            if text['response'].strip() == '':
                continue
            if text['response'] == '$EXIT$':
                break
            role = 1 if text['role'] == 'A' else -1
            doc = nlp(text['response'])
            embeddings_data: list[np.ndarray] = []
            for token in doc:
                if token.lemma_ in embeddings:
                    embeddings_data.append(embeddings[token.lemma_])
            embeddings_data = np.array(embeddings_data)
            dialogue_data.append((role, embeddings_data))
        if len(dialogue_data) > 0:
            data.append((dialogue_data, label))
    with open("features.pkl", "wb") as f:
        pickle.dump(data, f)
    return data

def load_data(file: str = "features.pkl") -> list[tuple[Dialog, Label]]:
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    preprocess_data()
