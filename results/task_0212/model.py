import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return F.l1_loss(input, target)


batch_size = 512
channels = 3
height = 512
width = 512


def get_inputs():
    input = torch.randn(batch_size, channels, height, width)
    target = torch.randn(batch_size, channels, height, width)
    return [input, target]


def get_init_inputs():
    return []