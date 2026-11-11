import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = torch.max(x, dim=1, keepdim=True)  # torch.max
        x = nn.ReLU()(x)  # nn.ReLU
        return x


batch_size = 128
channels = 3
height = 1024
width = 512


def get_inputs():
    # [batch_size, channels, height, width] with height dimension >=512
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    # No specific inputs needed for this case because the class doesn't need any initialization parameters
    return []