import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return torch.cat((hidden, cell), dim=1)


batch_size = 16
input_size = 100
hidden_size = 50
num_layers = 2
seq_len = 1024  # Increased from 512 to 1024 to increase runtime


def get_inputs():
    return [torch.randn(batch_size, seq_len, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, num_layers]