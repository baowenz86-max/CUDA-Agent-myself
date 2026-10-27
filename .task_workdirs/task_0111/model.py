import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            input_dim, input_dim, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )

    def forward(self, x):
        x = self.conv_transpose1d(x)
        x = torch.ge(x, 0)  # Element-wise greater than or equal to zero
        return x


batch_size = 16
input_dim = 256  # Increased from 64 to increase computation
height = 8192  # Increased from 512 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1


def get_inputs():
    return [torch.randn(batch_size, input_dim, height)]


def get_init_inputs():
    return [input_dim, kernel_size, stride, padding, output_padding, False]  # bias not needed