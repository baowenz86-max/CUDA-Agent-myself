import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, hard_shrink_lambd):
        super().__init__()
        self.hard_shrink_lambd = hard_shrink_lambd
        # 提取标量参数，契约要求传递标量 int
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size = kernel_size[0]
        else:
            self.kernel_size = kernel_size
        if isinstance(stride, (tuple, list)):
            self.stride = stride[0]
        else:
            self.stride = stride
        if isinstance(padding, (tuple, list)):
            self.padding = padding[0]
        else:
            self.padding = padding

    def forward(self, x):
        x = cuda_extension.hardshrink_forward(x, self.hard_shrink_lambd)
        x = cuda_extension.max_pool2d_forward(
            x, self.kernel_size, self.stride, self.padding
        )
        return x

    def debug_forward(self, x):
        x_shrink = cuda_extension.hardshrink_forward(x, self.hard_shrink_lambd)
        yield "step_001", x_shrink
        x_pool = cuda_extension.max_pool2d_forward(
            x_shrink, self.kernel_size, self.stride, self.padding
        )
        yield "step_002", x_pool
        yield "step_003", x_pool
