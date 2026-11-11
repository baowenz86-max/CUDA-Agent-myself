import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_channels, target_channels):
        super().__init__()
        self.target_channels = target_channels

    def forward(self, x1, x2, target):
        x = torch.cat((x1, x2), dim=1)
        loss = F.l1_loss(x, target)
        return loss


batch_size = 512
input_channels = 3
target_channels = 6
height, width = 256, 256  # Increased from 32x32 to 256x256


def get_inputs():
    x1 = torch.randn(batch_size, input_channels, height, width)
    x2 = torch.randn(batch_size, input_channels, height, width)
    target = torch.randn(batch_size, target_channels, height, width)
    return [x1, x2, target]


def get_init_inputs():
    return [input_channels, target_channels]