import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias_weight, bias_lstm, clamp_min, clamp_max):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias_weight)
        self.bias = nn.Parameter(torch.tensor(bias_lstm))
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = F.layer_norm(x, x.shape[-1:])
        x = self.layer_norm(x)
        x = torch.nan_to_num(x)
        x, (hn, cn) = self.lstm(x)
        x = x + self.bias
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x


input_size = 64
hidden_size = 32
num_layers = 2
bias_weight = True
bias_lstm = 0.1
clamp_min = -0.5
clamp_max = 0.5
batch_size = 512
seq_length = 200  # Increased from 10 to 200 to increase computation


def get_inputs():
    return [torch.randn(seq_length, batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, num_layers, bias_weight, bias_lstm, clamp_min, clamp_max]