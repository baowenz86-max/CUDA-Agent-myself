import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.mean(x, dim=self.dim, keepdim=True)
        x = nn.Softmax(dim=-1)(x)
        return x


batch_size, channels, height, width = 512, 16, 64, 1280
dim = 1


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [dim]