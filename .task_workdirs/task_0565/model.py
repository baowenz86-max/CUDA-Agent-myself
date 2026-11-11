import torch


class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = torch.abs(x)
        x = torch.clamp(x, min=self.min_value, max=self.max_value)
        return x


batch_size = 2048
channels = 16
height = 512
width = 28


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [0.0, 1.0]