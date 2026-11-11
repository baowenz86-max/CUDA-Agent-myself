import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, gather_dim, sort_dim):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.gather_dim = gather_dim
        self.sort_dim = sort_dim

    def forward(self, x, indices, sort_values):
        x = self.conv_transpose(x)
        x = torch.gather(x, self.gather_dim, indices)
        x, _ = torch.sort(x, dim=self.sort_dim)
        return x


batch_size = 550
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
gather_dim = 1
sort_dim = 1


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    indices = torch.randint(0, out_channels, (batch_size, 1, height * stride, width * stride))
    sort_values = torch.randn(batch_size, out_channels, height * stride, width * stride)
    return [x, indices, sort_values]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, gather_dim, sort_dim]