import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, num_features):
        super().__init__()
        self.conv_transpose_1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn_2d = nn.BatchNorm2d(num_features)

    def forward(self, x):
        x = self.conv_transpose_1d(x)
        x = x.unsqueeze(2)  # Add an extra dimension to match 2D batch norm
        x = self.bn_2d(x)
        x = x.squeeze(2)  # Remove the extra dimension
        return x


batch_size = 128
in_channels = 3
out_channels = 16
length = 8192  # Increased from 512 to 8192 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
num_features = out_channels


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, num_features]