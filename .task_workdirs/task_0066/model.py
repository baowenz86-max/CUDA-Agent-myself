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
        self.relu6 = nn.ReLU6()

    def forward(self, x, y):
        x = F.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            bias=self.conv_transpose1d.bias,
            stride=self.conv_transpose1d.stride,
            padding=self.conv_transpose1d.padding,
            output_padding=self.conv_transpose1d.output_padding,
        )
        mask = torch.ge(x, self.threshold)
        x = self.relu6(x)
        x = x * mask.float()
        return x


batch_size = 512
in_channels = 3
out_channels = 16
length = 2048  # Increased from 32 to 2048 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
threshold = 0.5


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    y = torch.randn(batch_size, in_channels, length)
    return [x, y]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, threshold]