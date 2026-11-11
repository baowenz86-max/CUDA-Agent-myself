import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # 保留参数结构，避免 state_dict 差异
        self.avg_pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = cuda_extension.avg_pool2d_forward(
            x,
            kernel_size=self.avg_pool.kernel_size,
            stride=self.avg_pool.stride,
            padding=self.avg_pool.padding
        )
        x = cuda_extension.min_dim_forward(x, dim=1)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x_pool = cuda_extension.avg_pool2d_forward(
            x,
            kernel_size=self.avg_pool.kernel_size,
            stride=self.avg_pool.stride,
            padding=self.avg_pool.padding
        )
        yield "step_001", x_pool
        result = cuda_extension.min_dim_forward(x_pool, dim=1)
        yield "step_002", result
