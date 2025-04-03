import ast
import pandas as pd
import pickle
import torch
from transformers import BertTokenizer, BertModel

from preprocess import Role, Text, Dialog, Label

def preprocess_data(out: str = "features-0/bert.pkl", file: str = "Data/train.csv") -> list[tuple[Dialog, Label]]:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    df = pd.read_csv(file)
    data: list[tuple[Dialog, Label]] = []
    for row in df.iloc:
        dialogue = ast.literal_eval(row['Dialogue'])['text']
        label = row['Label']
        dialogue_data: Dialog = []
        for text in dialogue:
            role = 1 if text['role'] == 'A' else -1
            if text['response'] == '$S$' or text['response'].strip() == '':
                dialogue_data.append((role, None))
                continue
            if text['response'] == '$EXIT$':
                break
            encoding = tokenizer(text['response'], return_tensors='pt')
            with torch.no_grad():
                outputs = model(**encoding)
            embeddings_data = outputs.last_hidden_state[0].detach().numpy()
            dialogue_data.append((role, embeddings_data))
        if len(dialogue_data) > 0:
            data.append((dialogue_data, label))
    with open(out, "wb") as f:
        pickle.dump(data, f)
    return data

def load_data(file: str) -> tuple[list[tuple[Dialog, Label]], int]:
    """
    Parameters
    ---
    file: str
        The file to load the data from.

    Returns
    ---
    tuple[list[tuple[Dialog, Label]], int]
        The data and the size of one word vector.
        Size of one word vector is by default 768
    """
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data, 768

if __name__ == '__main__':
    preprocess_data()
