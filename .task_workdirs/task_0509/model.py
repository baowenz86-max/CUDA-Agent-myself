import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.norm = nn.BatchNorm1d(input_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x


input_size = 8192
hidden_size = 4096


def get_inputs():
    x = torch.randn(256, input_size)
    return [x]


def get_init_inputs():
    return [input_size, hidden_size]