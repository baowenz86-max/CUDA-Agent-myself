import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.cummin(x, dim=0)[0]


def get_inputs():
    return [torch.randn(10000, 128)]  # Increased first dimension to 10000


def get_init_inputs():
    return []