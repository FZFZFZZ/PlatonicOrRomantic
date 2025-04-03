import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sys

from .lstm_advanced import load, device

class LstmBasic(nn.Module):
    def __init__(self,
                 vector_size: int,
                 small_hidden_size: int = 200,
                 small_output_size: int = 100,
                 large_hidden_size: int = 50):
        super().__init__()
        self.small_lstm = nn.LSTM(input_size=vector_size,
                                  hidden_size=small_hidden_size,
                                  batch_first=True)
        self.small_output = nn.Sequential(
            nn.Linear(small_hidden_size, small_output_size),
            nn.Sigmoid(),
        )
        self.V = small_output_size
        self.large_lstm = nn.LSTM(input_size=small_output_size,
                                  hidden_size=large_hidden_size,
                                  batch_first=True)
        self.large_output = nn.Linear(large_hidden_size, 3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # size of x: (N, L, l, v)
        N, L, l, v = x.shape
        small_lstm_inputs = x.reshape(N * L, l, v)  # size: (N * L, l, v)
        _, (small_lstm_output, _) = self.small_lstm(small_lstm_inputs)  # size: (1, N * L, h)
        small_outputs = self.small_output(small_lstm_output[0])  # size: (N * L, V)
        large_inputs = small_outputs.reshape(N, L, self.V)  # size: (N, L, V)
        _, (large_lstm_output, _) = self.large_lstm(large_inputs)  # size: (1, N, H)
        return self.large_output(large_lstm_output[0])  # size: (N, 3)


def train(X: torch.Tensor,
          y: torch.Tensor,
          vector_size: int,
          lr: float = 0.0001,
          epochs: int = 500,
          sample_size: int = 300
          ) -> nn.Module:
    """
    Parameters
    ---
    X: torch.Tensor
        The input data of size (N, L, l, v),
        where N is the number of dialogues,
        L is the number of texts per dialogue,
        l is the number of tokens per text,
        and v is the size of the token vector.
    y: torch.Tensor
        The target data of size (N,).
        The numbers are 0, 1, or 2.
    lr: float
        The learning rate.
    epochs: int
        The number of epochs to train the model.
    
    Returns
    ---
    nn.Module
        The trained model.
    """
    model = LstmBasic(vector_size).to(device)
    model = nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx].to(device)
        y_sample = y[idx].to(device)
        optimizer.zero_grad()
        y_pred = model(X_sample)
        loss = criterion(y_pred, y_sample)
        loss.backward()
        optimizer.step()

        y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
        y_true = y_sample.cpu().detach().numpy()
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Epoch: {epoch}, Loss: {loss.item()}, F1 Score: {f1}")
    return model

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m models.lstm_basic <feature file> <output file>")
        exit(1)
    file = sys.argv[1]
    out = sys.argv[2]
    torch.manual_seed(42)
    np.random.seed(42)
    X, y, vector_size = load(file)
    print(f"Data loaded with size: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train(X_train, y_train, vector_size)
    with torch.no_grad():
        y_pred = np.argmax(model(X_test.to(device)).cpu().detach().numpy(), axis=1)
    y_test = y_test.detach().numpy()
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"F1 score: {f1}")
    torch.save(model.state_dict(), out)


if __name__ == '__main__':
    main()
