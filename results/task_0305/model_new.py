import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, norm_type, kernel_size, stride):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return cuda_extension.lp_pool1d(x, self.norm_type, self.kernel_size, self.stride)

    def debug_forward(self, x):
        y = cuda_extension.lp_pool1d(x, self.norm_type, self.kernel_size, self.stride)
        yield "step_001", y


batch_size = 128
channels = 32
length = 16384
norm_type = 2
kernel_size = 3
stride = 2


def get_inputs():
    x = torch.randn(batch_size, channels, length)
    return [x]


def get_init_inputs():
    return [norm_type, kernel_size, stride]
