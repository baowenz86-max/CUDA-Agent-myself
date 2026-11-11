import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.custom_leaky_relu(x, 0.01)
        x = cuda_extension.custom_avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.custom_leaky_relu(x, 0.01)
        yield "step_001", x
        x = cuda_extension.custom_avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        yield "step_002", x
        yield "step_003", x
