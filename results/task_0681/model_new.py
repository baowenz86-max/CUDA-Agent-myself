import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, num_channels):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels

    def forward(self, x, y):
        x = cuda_extension.cat_forward(x, y, 1)
        x = cuda_extension.multiply_scalar_forward(x, self.input_size)
        return x

    def debug_forward(self, x, y):
        x = cuda_extension.cat_forward(x, y, 1)
        yield "step_001", x
        x = cuda_extension.multiply_scalar_forward(x, self.input_size)
        yield "step_002", x
        yield "step_003", x
