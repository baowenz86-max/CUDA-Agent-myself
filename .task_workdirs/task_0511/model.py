import torch


class Model(torch.nn.Module):
    def __init__(self, parameter_shape):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.randn(*parameter_shape))

    def forward(self, x):
        return x + self.parameter


def get_inputs():
    return [torch.randn(16, 3, 4096, 4096)]


def get_init_inputs():
    return [tuple([16, 3, 1, 1])]