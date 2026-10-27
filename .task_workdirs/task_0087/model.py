import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_groups, num_channels, input_size, hidden_size, num_layers, batch_first):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)

    def forward(self, x, h0, c0):
        x = self.group_norm(x)
        x = x.permute(0, 2, 1)  # Rearrange for LSTM (batch, seq, feature)
        x, _ = self.lstm(x, (h0, c0))
        x = x.permute(0, 2, 1)  # Rearrange back
        x = torch.div(x, 2.0)  # Example division
        return x


batch_size = 512  # Increased to ensure input dimension >=512
num_groups = 4
num_channels = 64
time_steps = 20  # Increased sequence length to increase runtime
input_size = num_channels
hidden_size = 128
num_layers = 2
batch_first = True


def get_inputs():
    x = torch.randn(batch_size, num_channels, time_steps)
    h0 = torch.randn(num_layers, batch_size, hidden_size)
    c0 = torch.randn(num_layers, batch_size, hidden_size)
    return [x, h0, c0]


def get_init_inputs():
    return [num_groups, num_channels, input_size, hidden_size, num_layers, batch_first]