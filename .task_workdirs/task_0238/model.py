import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, x, target):
        x = self.conv_transpose(x)
        x = torch.log2(torch.clamp(x, min=1e-7))  # Ensure input to log2 is positive
        loss = self.mse_loss(x, target)
        return loss


batch_size = 512  # Increased from 128 to 512 to ensure first dimension is >=512
in_channels = 3
out_channels = 16
length = 2048  # Increased from 32 to 2048 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    target = torch.randn(batch_size, out_channels, length * stride)
    return [x, target]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding]