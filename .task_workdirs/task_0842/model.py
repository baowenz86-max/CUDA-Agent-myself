import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x, target):
        x = self.conv_transpose(x)
        loss = F.binary_cross_entropy_with_logits(x, target)
        return loss


batch_size = 512
in_channels = 3
out_channels = 1
height, width = 64, 64  # Increased from 32x32 to 64x64
kernel_size = 5  # Increased from 3 to 5
stride = 2
padding = 2  # Adjusted from 1 to 2 to maintain output size
output_padding = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    target = torch.randn(batch_size, out_channels, height * stride, width * stride)
    return [x, target]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]