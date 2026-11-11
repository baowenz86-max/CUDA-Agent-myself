import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = F.log_softmax(x, dim=self.input_size)
        x = torch.tanh(x)
        return x


def get_inputs():
    return [torch.randn(32, 2048, 2048)]


def get_init_inputs():
    return [2]