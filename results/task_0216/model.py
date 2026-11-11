import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(x)
        x = F.hardswish(x)
        return x


batch_size = 512
channels = 16
height = 192
width = 192


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []