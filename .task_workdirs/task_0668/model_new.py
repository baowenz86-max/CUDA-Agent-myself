import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_classes, divisor):
        super().__init__()
        self.divisor = divisor

    def forward(self, x):
        x = cuda_extension.log_softmax_dim1(x)
        x = cuda_extension.scalar_div(x, self.divisor)
        return x

    def debug_forward(self, x):
        x = cuda_extension.log_softmax_dim1(x)
        yield 'step_001', x
        x = cuda_extension.scalar_div(x, self.divisor)
        yield 'step_002', x
        yield 'step_003', x


batch_size = 64
num_classes = 10
height = 1024
width = 256
divisor = 0.5


def get_inputs():
    x = torch.randn(batch_size, num_classes, height, width)
    return [x]


def get_init_inputs():
    return [num_classes, divisor]
