import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_features, eps):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(input_features, eps=eps)

    def forward(self, x):
        x = torch.sigmoid(x)
        x = self.batch_norm(x)
        return x


batch_size = 512
channels = 3
height, width = 128, 128


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [3, 1e-5]