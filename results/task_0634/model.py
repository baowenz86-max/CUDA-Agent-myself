import torch


class Model(torch.nn.Module):
    def __init__(self, inplace):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        mean = torch.mean(x, dim=[1, 2, 3], keepdim=True)
        x = torch.where(x < mean, torch.zeros_like(x), x)
        return x


batch_size = 512
channels = 3
height = 256
width = 256


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [False]  # since inplace is a boolean flag, we provide False as an argument