import torch
from torch import nn
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from preprocess import load_data, Dialog, Label

class LstmAdvanced(nn.Module):
    def __init__(self,
                 vector_size: int = 50,
                 small_sequence_size: int = 5,
                 small_hidden_size: int = 200,
                 small_output_size: int = 100,
                 large_sequence_size: int = 5,
                 large_hidden_size: int = 50):
        super().__init__()
        self.small_conv1d = nn.Conv1d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=small_sequence_size,
                                      padding=small_sequence_size - 1)
        self.small_lstm = nn.LSTM(input_size=vector_size,
                                  hidden_size=small_hidden_size,
                                  batch_first=True)
        self.small_output = nn.Sequential(
            nn.Linear(small_hidden_size, small_output_size),
            nn.Sigmoid(),
        )
        self.large_conv1d = nn.Conv1d(in_channels=1,
                                      out_channels=1,
                                      kernel_size=large_sequence_size,
                                      padding=large_sequence_size - 1)
        self.large_lstm = nn.LSTM(input_size=small_output_size,
                                  hidden_size=large_hidden_size,
                                  batch_first=True)
        self.large_output = nn.Sequential(
            nn.Linear(large_hidden_size, 3)
        )
    
    def forward(self, x: Dialog) -> torch.Tensor:
        large_inputs = []  # size: L, V
        for text in x:
            _, embeddings = text
            if embeddings.shape == (0,):
                continue
            small_inputs = torch.tensor(embeddings).T.unsqueeze(dim=1)  # size: (v, 1, l)
            small_lstm_inputs = self.small_conv1d(small_inputs).squeeze().T  # size: (l', v)
            _0, (small_lstm_output, _1) = self.small_lstm(small_lstm_inputs)  # size: (1, h)
            small_lstm_output = small_lstm_output[0]  # size: h
            small_output = self.small_output(small_lstm_output)  # size: V
            large_inputs.append(small_output)
        
        if len(large_inputs) == 0:
            print(x)
        large_inputs = torch.stack(large_inputs).T.unsqueeze(dim=1)  # size: (V, 1, L)
        large_lstm_inputs = self.large_conv1d(large_inputs).squeeze().T  # size: (L', V)
        _0, (large_lstm_output, _1) = self.large_lstm(large_lstm_inputs)  # size: (1, H)
        large_lstm_output = large_lstm_output[0]  # size: H
        return self.large_output(large_lstm_output)  # size: 3


def train(X: list[Dialog], y: list[Label], lr: float = 0.01, epochs: int = 2000) -> nn.Module:
    y = torch.tensor(y) + 1
    model = LstmAdvanced()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        rand_idx = random.randint(0, len(X) - 1)
        x = X[rand_idx]
        y_pred = model(x)
        y_true = y[rand_idx]
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
    return model

def load():
    data = load_data()
    X: list[Dialog] = []
    y: list[Label] = []
    for dialog, label in data:
        if len(dialog) == 0:
            continue
        X.append(dialog)
        y.append(label)
    return X, y

def main():
    X, y = load()
    print(f"Data loaded with size: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train(X_train, y_train)
    y_pred = []
    for x in X_test:
        y_pred.append(model(x).detach().numpy().argmax() - 1)
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))


if __name__ == '__main__':
    main()
