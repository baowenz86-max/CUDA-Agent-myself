import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self, num_classes, in_channels, out_channels, kernel_size, stride, padding, output_padding, scale_factor
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.num_classes = num_classes

    def forward(self, x, target):
        x = self.conv_transpose(x)
        x = self.upsample(x)
        target = F.one_hot(target, num_classes=self.num_classes)
        return x, target


batch_size = 1024
num_classes = 10
in_channels = 3
out_channels = 16
height, width = 24, 24  # Increased from 16x16 to 24x24
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
scale_factor = 2


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    target = torch.randint(0, num_classes, (batch_size, height * scale_factor, width * scale_factor))
    return [x, target]


def get_init_inputs():
    return [num_classes, in_channels, out_channels, kernel_size, stride, padding, output_padding, scale_factor]