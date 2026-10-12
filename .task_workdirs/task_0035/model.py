import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, parameter_dict_size, scale_factor, mode, upsample_kernel_size, upsample_padding):
        super().__init__()
        self.parameter_dict = nn.ParameterDict(parameter_dict_size)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.upsample_kernel_size = upsample_kernel_size
        self.upsample_padding = upsample_padding

    def forward(self, x):
        x = self.upsample(x)
        x = torch.logical_and(x > 0, x < 1)
        return x


batch_size = 16
channels = 3
height = 512  # Increased from 64 to 512 to increase computation
width = 1024  # Increased from 512 to 1024 to increase computation
parameter_dict_size = {"weights": torch.Tensor(channels, 1, 1)}
scale_factor = 2.0
mode = "bilinear"
upsample_kernel_size = (3, 3)
upsample_padding = (1, 1)


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [parameter_dict_size, scale_factor, mode, upsample_kernel_size, upsample_padding]