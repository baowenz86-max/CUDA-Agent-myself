import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        kernel_h, kernel_w = self.max_pool.kernel_size
        stride_h, stride_w = (self.max_pool.stride, self.max_pool.stride) if isinstance(self.max_pool.stride, int) else self.max_pool.stride
        padding_h, padding_w = (self.max_pool.padding, self.max_pool.padding) if isinstance(self.max_pool.padding, int) else self.max_pool.padding
        pooled, indices = cuda_extension.max_pool2d_with_indices_forward(
            x, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w
        )
        return pooled

    def debug_forward(self, x):
        kernel_h, kernel_w = self.max_pool.kernel_size
        stride_h, stride_w = (self.max_pool.stride, self.max_pool.stride) if isinstance(self.max_pool.stride, int) else self.max_pool.stride
        padding_h, padding_w = (self.max_pool.padding, self.max_pool.padding) if isinstance(self.max_pool.padding, int) else self.max_pool.padding
        pooled, indices = cuda_extension.max_pool2d_with_indices_forward(
            x, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w
        )
        yield 'step_001', (pooled, indices)
        yield 'step_002', pooled
