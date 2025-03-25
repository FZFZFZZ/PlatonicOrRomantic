import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np

from preprocess import load_data, Dialog, Label

VECTOR_SIZE = 50
MICRO_SEQUENCE_LENGTH = 40
MACRO_SEQUENCE_LENGTH = 25

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
        self.V = small_output_size
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # size of x: (N, L, l, v)
        N, L, l, v = x.shape
        small_inputs = x.transpose(2, 3).reshape(N * L * v, l).unsqueeze(dim=1)  # size: (N * L * v, 1, l)
        small_lstm_inputs = self.small_conv1d(small_inputs).squeeze()  # size: (N * L * v, l')
        small_lstm_inputs = small_lstm_inputs.reshape(N * L, v, -1).transpose(1, 2)  # size: (N * L, l', v)
        _, (small_lstm_output, _) = self.small_lstm(small_lstm_inputs)  # size: (1, N * L, h)
        small_lstm_output = small_lstm_output[0]  # size: (N * L, h)
        small_outputs = self.small_output(small_lstm_output)  # size: (N * L, V)
        large_inputs = small_outputs.reshape(N, L, self.V).transpose(1, 2).reshape(N * self.V, L).unsqueeze(dim=1)  # size: (N * V, 1, L)
        large_lstm_inputs = self.large_conv1d(large_inputs).squeeze().reshape(N, self.V, -1).transpose(1, 2)  # size: (N, L', V)
        _, (large_lstm_output, _) = self.large_lstm(large_lstm_inputs)  # size: (1, N, H)
        large_lstm_output = large_lstm_output[0]  # size: (N, H)
        return self.large_output(large_lstm_output)  # size: (N, 3)


def train(X: torch.Tensor, y: torch.Tensor, lr: float = 0.001, epochs: int = 100) -> nn.Module:
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
    model = LstmAdvanced()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        y_pred = torch.argmax(y_pred, dim=1).detach().numpy()
        y_true = y.detach().numpy()
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Epoch: {epoch}, Loss: {loss.item()}, F1 Score: {f1}")
    return model

def load():
    data = load_data()
    dialogues = [dialogue for dialogue, _ in data]
    X = []
    for dialogue in dialogues:
        dialogue_data = []
        for text in dialogue:
            _, embeddings = text
            l_0, v = embeddings.shape
            assert 0 < l_0 <= MICRO_SEQUENCE_LENGTH and v == VECTOR_SIZE
            embeddings = np.vstack((np.zeros((MICRO_SEQUENCE_LENGTH - l_0, v)), embeddings))
            dialogue_data.append(embeddings)
        dialogue_data = np.array(dialogue_data)
        L_0, l, v = dialogue_data.shape
        assert 0 < L_0 <= MACRO_SEQUENCE_LENGTH and l == MICRO_SEQUENCE_LENGTH and v == VECTOR_SIZE
        dialogue_data = np.vstack((np.zeros((MACRO_SEQUENCE_LENGTH - L_0, l, v)), dialogue_data))
        X.append(dialogue_data)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor([label for _, label in data]) + 1
    return X, y

def main():
    X, y = load()
    print(f"Data loaded with size: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train(X_train, y_train)
    y_pred = np.argmax(model(X_test).detach().numpy(), axis=1)
    y_test = y_test.detach().numpy()
    print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
    torch.save(model.state_dict(), "models/lstm_advanced.pth")


if __name__ == '__main__':
    main()
