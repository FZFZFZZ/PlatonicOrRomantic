import ast
import pandas as pd
import pickle
import torch
from transformers import BertTokenizer, BertModel

from preprocess import Role, Text, Dialog, Label

def preprocess_data(out: str = "features/bert.pkl", file: str = "Data/train.csv") -> list[tuple[Dialog, Label]]:
    df = pd.read_csv(file)
    all_texts: list[str] = []
    all_roles: list[list[Role]] = []
    labels: list[Label] = []
    for row in df.iloc:
        dialogue = ast.literal_eval(row['Dialogue'])['text']
        texts: list[str] = []
        roles: list[Role] = []
        for text in dialogue:
            if text['response'] == '$S$':
                continue
            if text['response'].strip() == '':
                continue
            if text['response'] == '$EXIT$':
                break
            texts.append(text['response'])
            role = 1 if text['role'] == 'A' else -1
            roles.append(role)
        if len(texts) == 0:
            continue
        all_texts.extend(texts)
        all_roles.append(roles)
        labels.append(row['Label'])
    print("All texts loaded,", len(labels))
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    encoding = tokenizer.batch_encode_plus(
        all_texts,
        add_special_tokens=True,
        padding=True,
        return_tensors='pt',
    )
    with torch.no_grad():
        outputs = model(**encoding)
    embeddings = outputs.last_hidden_state.numpy()
    print("Embeddings calculated,", embeddings.shape)
    assert len(embeddings) == len(all_texts)
    data: list[tuple[Dialog, Label]] = []
    idx = 0
    for roles, label in zip(all_roles, labels):
        texts: list[Text] = []
        for role in roles:
            texts.append((role, embeddings[idx]))
            idx += 1
        data.append((texts, label))
    print("Done,", len(data))
    assert idx == len(embeddings)

    with open(out, "wb") as f:
        pickle.dump(data, f)
    return data

if __name__ == '__main__':
    preprocess_data()
