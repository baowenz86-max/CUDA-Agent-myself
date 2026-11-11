import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = F.gelu(x)
        x = torch.softmax(x, dim=1)
        return x


batch_size = 512


def get_inputs():
    return [torch.randn(batch_size, 1000, 128)]


def get_init_inputs():
    return [128, 64]