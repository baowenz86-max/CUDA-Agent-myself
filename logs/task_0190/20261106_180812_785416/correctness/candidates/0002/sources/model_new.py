import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = cuda_extension.prod_channel_keepdim(x)
        kh, kw = self.kernel_size
        sh = sw = self.stride if isinstance(self.stride, int) else self.stride[0]
        ph, pw = self.padding
        x = cuda_extension.max_pool2d(x, kh, kw, sh, ph, pw)
        return x

    def debug_forward(self, x):
        x = cuda_extension.prod_channel_keepdim(x)
        yield 'step_001', x
        kh, kw = self.kernel_size
        sh = sw = self.stride if isinstance(self.stride, int) else self.stride[0]
        ph, pw = self.padding
        x = cuda_extension.max_pool2d(x, kh, kw, sh, ph, pw)
        yield 'step_002', x
        yield 'step_003', x
