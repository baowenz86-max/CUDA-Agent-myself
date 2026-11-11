import torch


class Model(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        x = torch.sum(x, dim=self.in_channels)
        x = torch.expm1(x)
        return x


batch_size = 512
channels = 3
height = 512
width = 512


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [1]  # The dimension along which the sum is performed