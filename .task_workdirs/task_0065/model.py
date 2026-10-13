import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.subtract_mean = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        return x - mean - self.subtract_mean


def get_init_inputs():
    return []


def get_inputs():
    batch_size = 64
    channels = 3
    height = 1024
    width = 1024
    x = torch.randn(batch_size, channels, height, width)
    return [x]