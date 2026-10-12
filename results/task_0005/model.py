import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = torch.log10(x)
        x = torch.lt(x, 0.5)
        x = self.conv_transpose1d(x.float())  # Convert boolean tensor to float tensor before passing to ConvTranspose1d
        x = x + self.bias
        return x


batch_size = 128
in_channels = 64
out_channels = 32
length = 4096  # Increased from 1024 to increase computation time
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1)


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]