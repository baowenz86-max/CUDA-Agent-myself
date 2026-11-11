import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, target_size, reduction):
        super().__init__()
        self.pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.target_size = target_size

    def forward(self, x, target, mask1, mask2):
        x = self.pad(x)
        x = self.conv(x)
        loss = self.criterion(x, target)
        mask = torch.logical_or(mask1, mask2)
        return x, loss, mask


batch_size = 32
in_channels = 3
out_channels = 10
height, width = 512, 256  # Increased width from 32 to 256
kernel_size = 3
padding = 1
target_size = (batch_size, height, width)
reduction = "mean"


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    target = torch.randint(0, out_channels, (batch_size, height, width))
    mask1 = torch.randint(0, 2, (batch_size, height, width), dtype=torch.bool)
    mask2 = torch.randint(0, 2, (batch_size, height, width), dtype=torch.bool)
    return [x, target, mask1, mask2]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, padding, target_size, reduction]