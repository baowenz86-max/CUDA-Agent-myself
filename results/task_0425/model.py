import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, window_size):
        super(Model, self).__init__()
        self.window_size = window_size

    def forward(self, x):
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x_triu = torch.triu(x, diagonal=0)
        return x_max, x_triu


batch_size = 512
input_dim = 32
window_size = 100  # Increased window size to increase computation


def get_inputs():
    x = torch.randn(batch_size, input_dim, window_size, window_size)
    return [x]


def get_init_inputs():
    return [input_dim, window_size]