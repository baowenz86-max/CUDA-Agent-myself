import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features

    def forward(self, x):
        x = torch.mul(x, torch.sign(self.input_features))
        return x


batch_size = 64
seq_len = 2048
feature_size = 2048


def get_inputs():
    return [torch.randn(batch_size, seq_len, feature_size)]


def get_init_inputs():
    return [torch.randn(feature_size)]