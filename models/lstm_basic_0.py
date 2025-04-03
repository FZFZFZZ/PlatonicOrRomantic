import numpy as np
import torch
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from .lstm_advanced import device
from .lstm_advanced_0 import load
from .lstm_basic import train

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m models.lstm_basic_0 <feature file> <output file>")
        exit(1)
    file = sys.argv[1]
    out = sys.argv[2]
    torch.manual_seed(42)
    np.random.seed(42)
    X, y, vector_size = load(file)
    print(f"Data loaded with size: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train(X, y, vector_size)
    with torch.no_grad():
        y_pred = np.argmax(model(X_test.to(device)).cpu().detach().numpy(), axis=1)
    y_test = y_test.detach().numpy()
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"F1 score: {f1}")
    torch.save(model.state_dict(), out)


if __name__ == '__main__':
    main()
