import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_features, eps=eps)

    def forward(self, x):
        return torch.sum(self.layer_norm(x), dim=1)


num_features = 128
batch_size = 512  # Increased from 64 to increase computation
seq_length = 1024  # seq_length is already 1024, which is >=512


def get_inputs():
    x = torch.randn(batch_size, seq_length, num_features)
    return [x]


def get_init_inputs():
    return [num_features]