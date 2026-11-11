import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, norm_type, kernel_size, stride=None, ceil_mode=False):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return F.lp_pool2d(
            x, norm_type=self.norm_type, kernel_size=self.kernel_size, stride=self.stride, ceil_mode=self.ceil_mode
        )


batch_size = 128
channels = 3
height, width = 512, 512  # Increased width to 512
norm_type = 2
kernel_size = 3
stride = 2
ceil_mode = False


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [norm_type, kernel_size, stride, ceil_mode]