import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, ceil_mode=False):
        super().__init__()
        # Store configuration as plain attributes; no parameters or buffers.
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, x):
        kh, kw = self.kernel_size
        sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        ph, pw = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        x = cuda_extension.avg_pool2d(x, kh, kw, sh, sw, ph, pw, self.ceil_mode)
        x = cuda_extension.leaky_relu(x, 0.01)
        return x

    def debug_forward(self, *inputs):
        # Inputs tuple contains x only.
        x = inputs[0]
        kh, kw = self.kernel_size
        sh, sw = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        ph, pw = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        after_pool = cuda_extension.avg_pool2d(x, kh, kw, sh, sw, ph, pw, self.ceil_mode)
        yield 'step_001', after_pool
        after_relu = cuda_extension.leaky_relu(after_pool, 0.01)
        yield 'step_002', after_relu
        yield 'step_003', after_relu
