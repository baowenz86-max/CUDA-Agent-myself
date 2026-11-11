import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, norm_type, kernel_size, stride=None, ceil_mode=False):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return cuda_extension.lp_pool2d_forward(
            x,
            norm_type=float(self.norm_type),
            kernel_size=int(self.kernel_size),
            stride=self.stride,
            ceil_mode=bool(self.ceil_mode),
        )

    def debug_forward(self, x):
        result = cuda_extension.lp_pool2d_forward(
            x,
            norm_type=float(self.norm_type),
            kernel_size=int(self.kernel_size),
            stride=self.stride,
            ceil_mode=bool(self.ceil_mode),
        )
        yield "step_001", result