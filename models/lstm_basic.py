import ast
import pandas as pd
import numpy as np
from typing import Literal
import pickle
import sys

Role = Literal[-1, 1]
# array shape: (L, V) where L is number of tokens, V is vector size (currently 50)
# None means a long pause
Text = tuple[Role, np.ndarray | None]
Dialog = list[Text]
Label = Literal[-1, 0, 1]

def get_embeddings(file: str) -> dict:
    embeddings = {}
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split(" ")
            word = values[0]
            arr = [float(item) if item != "." else 0.0 for item in values[1:]]
            vector = np.array(arr)
            embeddings[word] = vector
    return embeddings

def load_data(file: str) -> tuple[list[tuple[Dialog, Label]], int]:
    """
    Paremeters
    ---
    file: str
        The file to load the data from.
    
    Returns
    ---
    tuple[list[tuple[Dialog, Label]], int]
        The data and the size of each word vector.
        Size of each word vector is inferred from the file name.
    """
    with open(file, "rb") as f:
        data = pickle.load(f)
    size = int(file.split(".")[-2][:-1])
    return data, size

from torch.nn.utils.rnn import PackedSequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
import torch.nn.init as init
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def to_inputs_and_labels(data: list[tuple[Dialog, Label]]) -> tuple[list[list[np.ndarray]], list[Label]]:
    '''Split data into inputs and labels

    Args:
        data(list[Dialog]): loaded training data, with non-uniform batch size
        and sequence length.  
                
    :Returns: tuple[inputs, labels] WHERE

        inputs(list[list[np.ndarray]]): input data, with non-uniform batch size
        and sequence length, ignoring Role.
        
        labels(list[Label]): labels for each dialog
    '''
    inputs = []
    labels = []
    for batch in data:
        inputs_batch = []
        labels.append(batch[1])
        for pair in batch[0]:
            inputs_batch.append(pair[1])
        inputs.append(inputs_batch)
    return (inputs, labels)


def collate_inputs(inputs: list[list[np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r'''Pad sequences to global max_seq_len. Then pad batches to max_batch_size.

    Use this function instead of calling nn.utils.rnn.pad_sequence on each batch,
    because each batch has its own max_seq_len, and we need to use the global max.

    Args:
        inputs(list[list[np.ndarray]]): All input data
    
    Returns:
        tuple(tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        Input tensor in the shape (num_batch, max_batch_size, max_seq_len),
        seq_lens tensor in the shape (num_batch, max_batch_size), and
        batch_sizes tensor in the shape (num_batch)
    '''
    # Flatten seqs across all dialogs and convert type from np.ndarray to torch.Tensor
    all_sequences = [torch.tensor(seq) for dialog in inputs for seq in dialog]

    max_seq_len = max(seq.size(0) for seq in all_sequences)
    padded_sequences = [F.pad(seq, (0, 0, 0, max_seq_len - seq.size(0))) for seq in all_sequences]

    # Group padded sequences back into dialog structure
    num_batch = len(inputs)
    max_batch_size = max(len(dialog) for dialog in inputs)

    padded_dialogs = []
    seq_lens = []
    batch_sizes = []

    idx = 0
    for dialog in inputs:
        padded_dialogs.append(torch.stack(padded_sequences[idx:idx + len(dialog)]))
        seq_lens.append(torch.tensor([len(seq) for seq in dialog]))
        batch_sizes.append(len(dialog))
        idx += len(dialog)

    # Pad dialogs and labels with zero sequences to match max_batch_size
    for i in range(num_batch):
        while len(padded_dialogs[i]) < max_batch_size:
            padded_dialogs[i] = torch.cat((padded_dialogs[i], torch.zeros(1, max_seq_len, padded_dialogs[i].size(-1))), dim=0)
            seq_lens[i] = torch.cat((seq_lens[i], torch.tensor([0])))

    return torch.stack(padded_dialogs), torch.stack(seq_lens), torch.tensor(batch_sizes)


def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:  # Input weights
                init.xavier_uniform_(param.data)
            elif "weight_hh" in name:  # Hidden weights
                init.orthogonal_(param.data)
            elif "bias" in name:  # Bias terms
                param.data.fill_(0)  # Zero bias to start with

    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


class LstmDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, X, y, seq_lens, batch_sizes):
        self.X = X
        self.y = y
        self.seq_lens = seq_lens
        self.batch_sizes = batch_sizes
       
    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (self.X, self.y, self.seq_lens, self.batch_sizes)
    
    def __len__(self):
        return len(self.y)


def load(path, input_dim):
    data = load_data(path)[0]

    # Data: list[tuple[Dialog, Label]]

    X, y = to_inputs_and_labels(data)
    X, seq_lens, batch_sizes = collate_inputs(X)
    y = torch.tensor(y)
    y = y + 1
    y = y.long()
    X = X.float()

    train_X, test_X, train_y, test_y, train_seq_lens, test_seq_lens, train_batch_sizes, test_batch_sizes = train_test_split(
        X, y, seq_lens, batch_sizes,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    train_data = LstmDataset(train_X, train_y, train_seq_lens, train_batch_sizes)
    test_data = LstmDataset(test_X, test_y, test_seq_lens, test_batch_sizes)
    return (train_data, test_data)


class TokenLSTM(nn.Module):
    '''Token LSTM Model

        This is a model which takes in token embeddings and generate a
        seq-aware token embedding for each sequence.

        X: input, shape: (num_batch, max_batch_size, max_seq_len, token_embed_size)
        
        pred_y: output, shape: (num_batch * max_batch_size, max_seq_len, hidden_size)
    '''
    def __init__(self, input_dim, hidden_dim, num_layers):
        '''
        Args:
            input_dim: (input size) token_embed_size
            hidden_dim: (output size) hidden_size
        '''
        super(TokenLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True).float()
    
    def _pack_wrapper(self, X: torch.Tensor, seq_lens: list) -> PackedSequence:
        '''Mask the padding in input tensor to improve computational efficiency.
        This is a wrapper to specify reshaping operations involved.

        Returns:
            packed_X(torch.nn.utils.rnn.PackedSequence): packed input tensor
        '''
        flattened_X = X.view(-1, X.size(-2), X.size(-1))
        flattened_seq_lens = np.array(seq_lens).reshape(-1)

        # Filter out zero-length sequences, since pack_padded_sequence() only
        # accepts non-empty sequences.
        non_empty_mask = flattened_seq_lens > 0
        filtered_X = flattened_X[non_empty_mask]
        filtered_seq_lens = flattened_seq_lens[non_empty_mask]
        
        # Pack the sequences
        return pack_padded_sequence(filtered_X, 
                                    filtered_seq_lens, 
                                    batch_first=True,
                                    enforce_sorted=False)

    def _reintroduce_empty_seqs(self, pred_y, X, seq_lens):
        augmented_y = torch.zeros((X.size(0) * X.size(1), X.size(2), self.hidden_dim), device='cuda')
        non_empty_mask = torch.tensor(np.array(seq_lens).reshape(-1) > 0, device='cuda')
        augmented_y[non_empty_mask] = pred_y
        return augmented_y

    def forward(self, X: torch.Tensor, seq_lens: torch.Tensor):
        packed_X = self._pack_wrapper(X, seq_lens)
        packed_y, _ = self.lstm(packed_X)
        pred_y, _ = pad_packed_sequence(packed_y, batch_first=True)
        return self._reintroduce_empty_seqs(pred_y, X, seq_lens)
    

class SequenceLSTM(nn.Module):
    '''Sequence LSTM Model

        This is a model which takes in seq-aware token embeddings obtained from
        TokenLSTM and generate a dialog-aware seq embedding for each seq.

        X: input, shape: (num_batch * max_batch_size, max_seq_len, token_lstm_hidden_size)
        
        pred_y: prediction for each seq, shape: (num_batch, max_batch_size, seq_embed_size)

        dialog_embed: output, embedding of the last sequence, shape: (num_batch, seq_embed_size)
    '''
    def __init__(self, input_dim, hidden_dim, num_layers):
        '''
        Args:
            input_dim: (input size) token_lstm_hidden_size
            hidden_dim: (output size) seq_embed_size
        '''
        super(SequenceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True).float()

    def forward(self, X, batch_sizes):
        packed_X = pack_padded_sequence(X, batch_sizes, batch_first=True, enforce_sorted=False)
        packed_y, _ = self.lstm(packed_X)
        pred_y, _ = pad_packed_sequence(packed_y, batch_first=True)
        dialog_embed = pred_y[:, -1, :]
        return dialog_embed


class BasicLSTM(nn.Module):
    def __init__(self, embed_size, h_1, h_2):
        super(BasicLSTM, self).__init__()
        self.token_lstm = TokenLSTM(embed_size, h_1, 1)
        self.sequence_lstm = SequenceLSTM(h_1, h_2, 1)
        self.fc = nn.Linear(h_2, 3)

    def forward(self, X, seq_lens, batch_sizes):
        X_1 = self.token_lstm(X, seq_lens)
        X_2 = self.sequence_lstm(X_1, batch_sizes)
        logits = self.fc(X_2)
        pred_y = F.log_softmax(logits, dim=-1)
        return pred_y




def train(
    model: BasicLSTM, 
    train_data: LstmDataset, 
    loss_function: nn.CrossEntropyLoss, 
    optimizer: torch.optim.Adam,
    num_epochs=5,
    device='cuda'
) -> None:
    model.train()

    for epoch in tqdm(range(num_epochs)):
        total_loss = 0.0

        # Loads the whole training dataset.
        # Index here can be any value. We have it here to call the
        # __getitem__ method of Dataset. Index value is not used.
        X, y, seq_lens, batch_sizes = train_data[0]
        X = X.to(device)
        y = y.to(device)
        seq_lens = seq_lens.tolist()
        batch_sizes = batch_sizes.tolist()

        y_pred = model.forward(X, seq_lens, batch_sizes).squeeze()

        optimizer.zero_grad()
        loss = loss_function(y_pred, y)
        loss.backward()

        # To avoid exploding gradients
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python lstm_basic.py <learning rate> <number of epochs>")
        exit(1)
    lr = float(sys.argv[1])
    num_epochs = int(sys.argv[2])
    data_path = 'data/glove.6B.50d.pkl'
    input_dim = 50
    train_data, test_data = load(data_path, input_dim)

    hidden_dim = 256
    device = 'cuda'

    model = BasicLSTM(input_dim, hidden_dim, hidden_dim).to(device)
    model.apply(init_weights)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(model, train_data, loss_function, optimizer, num_epochs=num_epochs, device=device)

    torch.save(model, f"lstm_basic_{sys.argv[1]}_{sys.argv[2]}.pt")


if __name__ == '__main__':
    main()
