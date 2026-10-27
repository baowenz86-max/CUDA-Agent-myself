import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        x = torch.exp(x)
        return x


seq_len = 1024  # Increased from 512 to double computation
batch_size = 16
input_size = 20
hidden_size = 30
num_layers = 2
num_classes = 1


def get_inputs():
    # The input to LSTM should be of shape: (seq_len, batch, input_size)
    input_data = torch.randn(seq_len, batch_size, input_size)
    return [input_data]


def get_init_inputs():
    return [input_size, hidden_size, num_layers, num_classes]