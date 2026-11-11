import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_pool_kernel_size):
        super().__init__()
        self.hardtanh = nn.Hardtanh()
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_kernel_size)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = cuda_extension.custom_hardtanh(x, -1.0, 1.0)
        kernel_h, kernel_w = self.max_pool.kernel_size
        stride_h, stride_w = self.max_pool.stride
        padding_h, padding_w = self.max_pool.padding
        dilation_h, dilation_w = self.max_pool.dilation
        ceil_mode = self.max_pool.ceil_mode
        x = cuda_extension.custom_max_pool2d(
            x,
            kernel_h,
            stride_h,
            padding_h,
            dilation_h,
            ceil_mode,
        )
        x = cuda_extension.custom_flatten(x, 1)
        x = cuda_extension.custom_linear(x, self.linear.weight, self.linear.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_hardtanh(x, -1.0, 1.0)
        yield "step_001", x
        kernel_h, kernel_w = self.max_pool.kernel_size
        stride_h, stride_w = self.max_pool.stride
        padding_h, padding_w = self.max_pool.padding
        dilation_h, dilation_w = self.max_pool.dilation
        ceil_mode = self.max_pool.ceil_mode
        x = cuda_extension.custom_max_pool2d(
            x,
            kernel_h,
            stride_h,
            padding_h,
            dilation_h,
            ceil_mode,
        )
        yield "step_002", x
        x = cuda_extension.custom_flatten(x, 1)
        yield "step_003", x
        x = cuda_extension.custom_linear(x, self.linear.weight, self.linear.bias)
        yield "step_004", x
        yield "step_005", x
