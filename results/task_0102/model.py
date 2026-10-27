import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = F.elu(x)
        x = torch.triu(x)
        return x


batch_size = 512
channels = 3
height = 256
width = 256


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []