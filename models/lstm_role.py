import torch
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sys

from .lstm_advanced import device

from preprocess import load_data as load_glove_data
from preprocess_bert import load_data as load_bert_data

from typing import Literal
Role = Literal[-1, 1]

# Length 40 for 6b.50d and 6b.100d, Length 50 for the rest of glove
# Length 75 for bert
MICRO_SEQUENCE_LENGTH = 40
MACRO_SEQUENCE_LENGTH = 25


class CombineLabel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # size of x: (N, 2, 2)
        p0 = x[:, :, 0]
        p1 = x[:, :, 1]
        return torch.stack([
            p0[:, 0] * p0[:, 1],
            p0[:, 0] * p1[:, 1] + p1[:, 0] * p0[:, 1],
            p1[:, 0] * p1[:, 1],
        ], dim=1)

class LstmBasic(nn.Module):
    def __init__(self,
                 vector_size: int,
                 small_hidden_size: int = 200,
                 small_output_size: int = 100,
                 large_hidden_size: int = 50,
                 role_count: int = 2):
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
        self.large_output = nn.Linear(large_hidden_size, 2)
        self.final_output = nn.Linear(role_count * 2, 3)
        self.combine_label = CombineLabel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # size of x: (N, r, L, l, v)
        N, r, L, l, v = x.shape
        small_lstm_inputs = x.reshape(N * r * L, l, v)  # size: (N * r * L, l, v)
        _, (small_lstm_output, _) = self.small_lstm(small_lstm_inputs)  # size: (1, N * r * L, h)
        small_outputs = self.small_output(small_lstm_output[0])  # size: (N * r * L, V)
        large_inputs = small_outputs.reshape(N * r, L, self.V)  # size: (N * r, L, V)
        _, (large_lstm_output, _) = self.large_lstm(large_inputs)  # size: (1, N * r, H)
        large_outputs = self.large_output(large_lstm_output[0])  # size: (N * r, 2)
        label_inputs = large_outputs.reshape(N, r, 2)
        return self.combine_label(label_inputs)

def train(X: torch.Tensor,
          y: torch.Tensor,
          vector_size: torch.Tensor,
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


def load(file: str):
    if file.endswith("bert.pkl"):
        data, vector_size = load_bert_data(file)
    else:
        data, vector_size = load_glove_data(file)
    
    dialogues = [dialogue for dialogue, _ in data]
    X = []
    
    for dialogue in dialogues:
        dialogue_data = [[],[]]
        for text in dialogue:
            role, embeddings = text
            if embeddings is None:
                l_0, v = (MICRO_SEQUENCE_LENGTH, vector_size)
                embeddings = np.zeros((MICRO_SEQUENCE_LENGTH, v))
            else:
                l_0, v = embeddings.shape
                assert 0 < l_0 <= MICRO_SEQUENCE_LENGTH and v == vector_size
            embeddings = np.vstack((np.zeros((MICRO_SEQUENCE_LENGTH - l_0, v)), embeddings))
            if role == -1:
                dialogue_data[0].append(embeddings)
            else:
                dialogue_data[1].append(embeddings)

        dialogue_data[0] = np.array(dialogue_data[0])
        dialogue_data[1] = np.array(dialogue_data[1])
        if dialogue_data[0].shape[0] != 0:
            L_0_0, l, v = dialogue_data[0].shape
            assert 0 < L_0_0 <= MACRO_SEQUENCE_LENGTH and l == MICRO_SEQUENCE_LENGTH and v == vector_size
        else:
            L_0_0, l, v = (MACRO_SEQUENCE_LENGTH, MICRO_SEQUENCE_LENGTH, vector_size)
            dialogue_data[0] = np.zeros((L_0_0, l, v))
        if dialogue_data[1].shape[0] != 0:
            L_0_1, l, v = dialogue_data[1].shape
            assert 0 < L_0_1 <= MACRO_SEQUENCE_LENGTH and l == MICRO_SEQUENCE_LENGTH and v == vector_size
        else:
            L_0_1, l, v = (MACRO_SEQUENCE_LENGTH, MICRO_SEQUENCE_LENGTH, vector_size)
            dialogue_data[1] = np.zeros((L_0_1, l, v))
        dialogue_data[0] = np.vstack((np.zeros((MACRO_SEQUENCE_LENGTH - L_0_0, l, v)), dialogue_data[0]))
        dialogue_data[1] = np.vstack((np.zeros((MACRO_SEQUENCE_LENGTH - L_0_1, l, v)), dialogue_data[1]))
        dialogue_data = np.array(dialogue_data)
        X.append(dialogue_data)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor([label for _, label in data]) + 1
    return X, y, vector_size


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
