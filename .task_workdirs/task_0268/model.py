import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, cat_dim):
        super().__init__()
        self.cat_dim = cat_dim

    def forward(self, x, y):
        x = torch.log_softmax(x, dim=self.cat_dim)
        y = torch.log_softmax(y, dim=self.cat_dim)
        return torch.cat((x, y), dim=self.cat_dim)


def get_inputs():
    x = torch.randn(8192, 512)
    y = torch.randn(8192, 512)
    return [x, y]


def get_init_inputs():
    return [0]