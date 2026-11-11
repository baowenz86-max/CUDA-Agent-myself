import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp

    def forward(self, x):
        x = F.softmax(x, dim=1)
        x, _ = torch.sort(x, dim=1, descending=True)
        return x


batch_size = 128
height, width = 2048, 32  # Increased height to increase computation


def get_inputs():
    return [torch.randn(batch_size, height, width)]


def get_init_inputs():
    return []