import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, norm_type, num_channels):
        super().__init__()
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(num_channels)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(num_channels)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(num_channels)
        else:
            raise ValueError("Invalid normalization type. Choose 'batch', 'instance', or 'layer'.")

    def forward(self, x):
        return self.norm(x)


batch_size = 32  # Increased from 16 to 32 to increase runtime
num_channels = 3
width = 512
height = 256
norm_type = "batch"


def get_inputs():
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]


def get_init_inputs():
    return [norm_type, num_channels]