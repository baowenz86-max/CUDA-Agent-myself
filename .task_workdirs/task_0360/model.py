import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)


batch_size = 16
input_size = 100
hidden_size = 64
num_layers = 2
bias = True
seq_length = 2048  # Increased sequence length to increase runtime


def get_inputs():
    x = torch.randn(seq_length, batch_size, input_size)
    return [x]


def get_init_inputs():
    return [input_size, hidden_size, num_layers, bias]