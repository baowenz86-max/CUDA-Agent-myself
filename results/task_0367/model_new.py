import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pooling_kernel):
        super().__init__()
        self.output_size_h = pooling_kernel[0]
        self.output_size_w = pooling_kernel[1]

    def forward(self, x):
        x = cuda_extension.custom_adaptive_avg_pool2d_forward(x, self.output_size_h, self.output_size_w)
        return x

    def debug_forward(self, *inputs):
        if len(inputs) != 1:
            raise TypeError("debug_forward expected 1 input tensor")
        x = inputs[0]
        x = cuda_extension.custom_adaptive_avg_pool2d_forward(x, self.output_size_h, self.output_size_w)
        yield "step_001", x
        yield "step_002", x
