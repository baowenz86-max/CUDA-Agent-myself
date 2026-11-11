import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = nn.LazyBatchNorm2d()

    def forward(self, x):
        x = self.batch_norm(x)
        return x


batch_size = 512
num_features = 32
height, width = 128, 128


def get_inputs():
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    return [num_features]