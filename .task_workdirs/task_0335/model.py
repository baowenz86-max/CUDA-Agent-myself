import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        out_channels,
        in_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        scale_factor,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(2)  # Add a temporal dimension
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv_transpose1d(x)
        return x


batch_size = 8192  # Increased from 512 to 8192 to increase runtime
in_features = 256
out_features = 128
out_channels = 64
in_channels = 128
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
scale_factor = 2


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [
        in_features,
        out_features,
        out_channels,
        in_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        scale_factor,
    ]