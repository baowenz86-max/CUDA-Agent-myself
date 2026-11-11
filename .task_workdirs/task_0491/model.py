import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.batch_norm_2d = nn.BatchNorm2d(in_features)

    def forward(self, x):
        x = self.batch_norm_2d(x)
        x = torch.clamp(x, min=0.0, max=1.0)
        return x


batch_size = 512
height, width, channels = 224, 224, 1


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [channels]