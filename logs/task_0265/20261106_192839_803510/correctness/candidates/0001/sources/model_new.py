import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_inputs, bias_shape):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, tensor1, tensor2):
        x = cuda_extension.fma_addcdiv(x, tensor1, tensor2, 1.0)
        x = cuda_extension.add_bias(x, self.bias, 3)
        return x

    def debug_forward(self, x, tensor1, tensor2):
        x = cuda_extension.fma_addcdiv(x, tensor1, tensor2, 1.0)
        yield 'step_001', x
        x = cuda_extension.add_bias(x, self.bias, 3)
        yield 'step_002', x
        yield 'step_003', x
