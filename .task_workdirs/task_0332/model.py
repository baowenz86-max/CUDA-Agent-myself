import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, pooling_length):
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(pooling_length)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * pooling_length, output_size)

    def forward(self, x):
        x = self.adaptive_avg_pool(x)
        x = x.transpose(1, 2)  # LSTM expects (batch, seq, feature)
        x, _ = self.lstm(x)
        x = x.reshape(x.size(0), -1)  # Flatten the LSTM output
        x = self.fc(x)
        return x


batch_size = 32
sequence_length = 512  # Modified to ensure at least one dimension >=512
input_size = 128
hidden_size = 256
num_layers = 2
output_size = 10
pooling_length = 64  # Increased from 16 to increase LSTM sequence length


def get_inputs():
    return [torch.randn(batch_size, input_size, sequence_length)]


def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, pooling_length]