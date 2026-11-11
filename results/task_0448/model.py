import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x_clone = torch.clone(x)
        return x_clone


batch_size = 64


def get_inputs():
    x = torch.randn(batch_size, 2048, 2048)
    return [x]


def get_init_inputs():
    return []