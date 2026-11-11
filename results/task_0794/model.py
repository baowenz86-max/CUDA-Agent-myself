import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, target):
        x = torch.logical_or(x > 0.5, y > 0.5)
        x = x.float()
        loss = F.mse_loss(x, target)
        return loss


batch_size = 512  # Adjusted to ensure at least one dimension (batch size) is >=512
in_channels = 3
height, width = 256, 256  # Increased spatial dimensions to increase computation


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    y = torch.randn(batch_size, in_channels, height, width)
    target = torch.randn(batch_size, in_channels, height, width)
    return [x, y, target]


def get_init_inputs():
    return []