import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, normalized_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = F.layer_norm(x, self.layer_norm.normalized_shape)
        x = torch.unbind(x, dim=1)
        return x


batch_size = 512
in_channels = 16
out_channels = 32
length = 2048  # Increased from 32 to 2048
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
normalized_shape = (out_channels, length * 2)  # Now (32, 4096)


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, normalized_shape]