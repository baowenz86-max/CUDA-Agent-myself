import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, x):
        x = F.log_softmax(x, dim=1)
        x = F.elu(x)
        return x


batch_size = 128
seq_len = 1024
in_features = 1024


def get_inputs():
    return [torch.randn(batch_size, seq_len, in_features)]


def get_init_inputs():
    return [in_features]