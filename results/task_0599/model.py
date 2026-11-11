import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, param_list):
        super().__init__()
        self.param_list = nn.ParameterList(param_list)

    def forward(self, x):
        x = torch.abs(x)
        for param in self.param_list:
            x = x + param
        return x


param_list = [nn.Parameter(torch.randn(3, 256, 256)), nn.Parameter(torch.randn(3, 256, 256))]
batch_size = 512  # Modified hyperparameter to ensure input has a dimension >=512


def get_inputs():
    return [torch.randn(batch_size, 3, 256, 256)]


def get_init_inputs():
    return [param_list]