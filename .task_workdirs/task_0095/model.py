import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, start, end, steps):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.start = start
        self.end = end
        self.steps = steps

    def forward(self, x):
        linear_space = torch.linspace(self.start, self.end, self.steps).to(x.device)
        x = torch.sub(x, linear_space)
        x = self.conv_transpose(x)
        return x


batch_size = 128
in_channels = 3
out_channels = 16
length = 16384  # Increased from 512 to 16384 to increase runtime
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
start = 0.0
end = 1.0
steps = length  # Steps follows the increased length


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, start, end, steps]