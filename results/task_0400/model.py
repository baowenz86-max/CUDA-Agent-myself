import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        x, _ = torch.max(x, dim=1, keepdim=True)  # Assuming we want to apply torch.max over the channel dimension
        x = self.relu6(x)
        return x


batch_size = 512
in_channels = 16
height = 256
width = 256


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return []