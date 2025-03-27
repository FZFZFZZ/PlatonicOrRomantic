import ast
import pandas as pd
import numpy as np
import spacy
from typing import Literal
import pickle
import sys

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

def preprocess_data(embeddings_file: str, out: str, file: str = "Data/train.csv") -> list[tuple[Dialog, Label]]:
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
            # This may cause issues because we are ignoring sentences only with emojis
            if len(embeddings_data) == 0:
                continue
            embeddings_data = np.array(embeddings_data)
            dialogue_data.append((role, embeddings_data))
        if len(dialogue_data) > 0:
            data.append((dialogue_data, label))
    with open(out, "wb") as f:
        pickle.dump(data, f)
    return data

def load_data(file: str) -> list[tuple[Dialog, Label]]:
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m preprocess <embeddings file> <output file>")
        exit(1)
    embeddings = sys.argv[1]
    out = sys.argv[2]
    preprocess_data(embeddings, out)

if __name__ == '__main__':
    main()
