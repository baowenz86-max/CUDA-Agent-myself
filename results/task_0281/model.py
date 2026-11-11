import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(x)
        x = self.avg_pool(x)
        return x


batch_size = 128
sequence_length = 10240  # Increased from 1024 to 10240
in_features = 64


def get_inputs():
    return [torch.randn(batch_size, in_features, sequence_length)]


def get_init_inputs():
    return []