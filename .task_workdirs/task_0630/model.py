import torch


class Model(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = torch.abs(x)
        x = torch.einsum("bixy->biy", x)
        return x


batch_size = 512
channels = 3
height = 512
width = 512


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [get_inputs()[0].size()]