import torch
import torch.nn as nn


class Model(nn.Module):  
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.bias
        x = torch.std(x, dim=[0, 2], unbiased=True)  # std over batch and width dimension
        return x


batch_size = 64
in_channels = 3
out_channels = 16
sequence_length = 16384  # Increased from 1024 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, sequence_length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]