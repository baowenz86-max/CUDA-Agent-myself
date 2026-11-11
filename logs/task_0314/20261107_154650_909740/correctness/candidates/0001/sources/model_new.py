import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # Store configuration only; no learnable parameters, state_dict stays empty.
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Ensure input is on CUDA as required by contract execution_device=cuda.
        if not x.is_cuda:
            x = x.cuda()
        # Step 1: LeakyReLU with negative_slope=0.01
        x = cuda_extension.custom_leaky_relu(x, 0.01)
        # Step 2: AvgPool2d with kernel_size, stride, padding
        x = cuda_extension.custom_avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        # step_001: leaky_relu
        x = cuda_extension.custom_leaky_relu(x, 0.01)
        yield "step_001", x
        # step_002: avg_pool
        x = cuda_extension.custom_avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        yield "step_002", x
        # step_003: return
        yield "step_003", x
