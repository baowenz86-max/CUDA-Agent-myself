import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        x = torch.remainder(x, 2)
        return x


batch_size = 512
channels = 128
height = 128
width = 128


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []