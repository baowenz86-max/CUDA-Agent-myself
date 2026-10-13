import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # Ensure input is on CUDA as required by execution device
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias, self.conv.stride[0], self.conv.padding[0])
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.maxpool2d_forward(x, self.pool.kernel_size, self.pool.stride, self.pool.padding, self.pool.dilation, self.pool.ceil_mode)
        return x

    def debug_forward(self, x):
        # Same computation, yielding after each step
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias, self.conv.stride[0], self.conv.padding[0])
        yield 'step_001', x
        x = cuda_extension.relu_forward(x)
        yield 'step_002', x
        x = cuda_extension.maxpool2d_forward(x, self.pool.kernel_size, self.pool.stride, self.pool.padding, self.pool.dilation, self.pool.ceil_mode)
        yield 'step_003', x
        yield 'step_004', x
