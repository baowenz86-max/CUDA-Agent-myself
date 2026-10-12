import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, threshold):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.threshold = threshold

    def forward(self, x, z):
        x = F.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            stride=self.conv_transpose1d.stride,
            padding=self.conv_transpose1d.padding,
            output_padding=self.conv_transpose1d.output_padding,
        )
        median_value = torch.median(x)
        x = torch.ne(x, median_value)
        z = torch.ne(z, self.threshold)
        return x, z


batch_size = 512
in_channels = 3
out_channels = 16
length = 512  # Increased from 32 to 512 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
threshold = 0.5


def get_inputs():
    return [torch.randn(batch_size, in_channels, length), torch.randn(batch_size, out_channels, length * stride)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, threshold]