import numpy as np
import torch
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from preprocess import load_data as load_glove_data
# Not using bert
# from preprocess_bert import load_data as load_bert_data
from .lstm_advanced import train, device

# Length 50 for glove
# Length 75 for bert
MICRO_SEQUENCE_LENGTH = 50
MACRO_SEQUENCE_LENGTH = 35

def load(file: str):
    # if file.endswith("bert.pkl"):
    #     data, vector_size = load_bert_data(file)
    # else:
    #     data, vector_size = load_glove_data(file)
    data, vector_size = load_glove_data(file)
    dialogues = [dialogue for dialogue, _ in data]
    X = []
    for dialogue in dialogues:
        dialogue_data = []
        for text in dialogue:
            _, embeddings = text
            if embeddings is None:
                dialogue_data.append(np.zeros((MICRO_SEQUENCE_LENGTH, vector_size)))
                continue
            l_0, v = embeddings.shape
            assert 0 < l_0 <= MICRO_SEQUENCE_LENGTH and v == vector_size
            embeddings = np.vstack((np.zeros((MICRO_SEQUENCE_LENGTH - l_0, v)), embeddings))
            dialogue_data.append(embeddings)
        dialogue_data = np.array(dialogue_data)
        L_0, l, v = dialogue_data.shape
        assert 0 < L_0 <= MACRO_SEQUENCE_LENGTH and l == MICRO_SEQUENCE_LENGTH and v == vector_size
        dialogue_data = np.vstack((np.zeros((MACRO_SEQUENCE_LENGTH - L_0, l, v)), dialogue_data))
        X.append(dialogue_data)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor([label for _, label in data]) + 1
    return X, y, vector_size

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m models.lstm_advanced_0 <feature file> <output file>")
        exit(1)
    file = sys.argv[1]
    out = sys.argv[2]
    torch.manual_seed(42)
    X, y, vector_size = load(file)
    print(f"Data loaded with size: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train(X_train, y_train, vector_size)
    with torch.no_grad():
        y_pred = np.argmax(model(X_test.to(device)).cpu().detach().numpy(), axis=1)
    y_test = y_test.detach().numpy()
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    torch.save(model.state_dict(), out)


if __name__ == '__main__':
    main()
