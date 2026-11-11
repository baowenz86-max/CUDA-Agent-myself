import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias_shape):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bias=True)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # LSTM expects input in the form (batch, seq, feature)
        x, (hn, cn) = self.lstm(x)
        x = torch.matmul(x, self.bias)
        x = torch.nan_to_num(x)
        return x


input_size = 10
hidden_size = 200  # Increased from 20 to increase LSTM complexity
num_layers = 2
batch_size = 512
seq_len = 500  # Increased from 5 to extend sequence processing
bias_shape = (hidden_size, input_size)


def get_inputs():
    return [torch.randn(batch_size, seq_len, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, num_layers, bias_shape]