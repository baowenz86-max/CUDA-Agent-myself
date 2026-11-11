import torch


class Model(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super(Model, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.var_mean(x, self.dim, self.keepdim)


def get_inputs():
    return [torch.randn(128, 64, 256, 256)]


def get_init_inputs():
    return [3, True]