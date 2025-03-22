import ast
import pandas as pd
import numpy as np
import spacy
from typing import Literal
import pickle

Text = tuple[Literal[-1, 1], np.ndarray]
Dialog = list[Text]

def get_embeddings(file: str) -> dict:
    embeddings = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings[word] = vector
    return embeddings

def preprocess_data(file: str = "Data/train.csv", embeddings_file: str = "vectors/glove.6B.50d.txt") -> list[Dialog]:
    embeddings = get_embeddings(embeddings_file)
    print("Embeddings loaded with size:", len(embeddings))
    nlp = spacy.load("en_core_web_sm")
    df = pd.read_csv(file)
    data: list[Dialog] = []
    for row in df.iloc:
        dialogue = ast.literal_eval(row['Dialogue'])['text']
        dialogue_data: Dialog = []
        for text in dialogue:
            if text['response'] == '$S$':
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
        data.append(dialogue_data)
    with open("features.pkl", "wb") as f:
        pickle.dump(data, f)
    return data

def load_data(file: str = "features.pkl") -> list[Dialog]:
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    preprocess_data()
