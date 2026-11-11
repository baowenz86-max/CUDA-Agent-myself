import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x_max, _ = torch.max(x, dim=1)
        x_stacked = torch.stack((x, x), dim=1)
        return x_max, x_stacked


batch_size = 512
channels = 8
height = 256
width = 256


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return []