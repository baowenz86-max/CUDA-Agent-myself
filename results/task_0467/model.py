import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = F.hardtanh(x)
        x = self.linear(x)
        return x


batch_size = 102400  # Increased from 512 to increase computation
input_features = 128


def get_inputs():
    return [torch.randn(batch_size, input_features)]


def get_init_inputs():
    return [input_features, 1024]