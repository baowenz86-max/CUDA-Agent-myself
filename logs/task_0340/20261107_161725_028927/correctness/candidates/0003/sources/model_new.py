import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.selu_forward(x)
        x = cuda_extension.cumsum0_forward(x)
        return x

    def debug_forward(self, *inputs):
        if len(inputs) != 1:
            raise ValueError('debug_forward expects exactly one input tensor')
        x = inputs[0]
        selu_out = cuda_extension.selu_forward(x)
        yield 'step_001', selu_out
        cumsum_out = cuda_extension.cumsum0_forward(selu_out)
        yield 'step_002', cumsum_out
        yield 'step_003', cumsum_out
