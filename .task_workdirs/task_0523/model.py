import torch


class Model(torch.nn.Module):
    def __init__(self, flip_dims):
        super().__init__()
        self.flip_dims = flip_dims

    def forward(self, x):
        return torch.flip(x, self.flip_dims)


batch_size = 16
channels = 3
height = 4096
width = 4096


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [(2,)]