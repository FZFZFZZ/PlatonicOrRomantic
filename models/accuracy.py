import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

from .lstm_advanced_0 import load
from .lstm_advanced import device
from .lstm_basic import LstmBasic

def main():
    X, y, v = load("features-0/glove.42B.300d.pkl")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = torch.nn.DataParallel(LstmBasic(v)).to(device)
    model.load_state_dict(torch.load("models/lstm_basic_0.glove.42B.300d.pth", weights_only=True))
    with torch.no_grad():
        y_pred = np.argmax(model(X_test.to(device)).cpu().detach().numpy(), axis=1)
    y_test = y_test.detach().numpy()
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"F1 score: {f1}")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    matrix = confusion_matrix(y_test, y_pred)
    matrix = matrix.diagonal() / matrix.sum(axis=1)
    print(f"Each class: {matrix}")


if __name__ == '__main__':
    main()
