import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_value, scale_factor, mode):
        super().__init__()
        self.prelu = nn.PReLU(in_channels, init=alpha_value)
        self.interpolate = F.interpolate
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.prelu(x)
        x = self.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = torch.argmax(x, dim=1, keepdim=True)
        return x


batch_size = 64
in_channels = 3
out_channels = 16
height, width = 512, 256  # Increased width to 256 to increase computation
kernel_size = 3
alpha_value = 0.25
scale_factor = 2.0
mode = "bilinear"


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, alpha_value, scale_factor, mode]