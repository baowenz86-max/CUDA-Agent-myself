import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, filter_size):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.filter_size = filter_size

    def forward(self, x, indices):
        x = self.conv_transpose1d(x)
        x = torch.gather(x, dim=1, index=indices)
        return x


batch_size = 128
in_channels = 64  # Increased from 3 to increase computation
out_channels = 16
length = 8192  # Increased from 512 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
# Calculate correct output length for ConvTranspose1d
output_length = (length - 1) * stride + kernel_size - 2 * padding + output_padding
filter_size = (batch_size, 1, output_length)  # Fixed filter size to match output length


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    indices = torch.randint(0, out_channels, filter_size)
    return [x, indices]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, filter_size]