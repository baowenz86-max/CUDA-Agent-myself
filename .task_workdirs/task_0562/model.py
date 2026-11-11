import torch


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = torch.abs(x)
        x = torch.round(x)
        return x


batch_size = 16
channels = 3
height = 4096  # Increased from 512 to 4096 to increase computation
width = 4096   # Increased from 224 to 4096 to increase computation


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []