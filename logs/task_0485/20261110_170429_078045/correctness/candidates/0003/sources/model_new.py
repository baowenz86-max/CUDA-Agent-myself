import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = cuda_extension.log_softmax_cuda(x, self.input_size)
        x = cuda_extension.tanh_cuda(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.log_softmax_cuda(x, self.input_size)
        yield ('step_001', x)
        x = cuda_extension.tanh_cuda(x)
        yield ('step_002', x)
        yield ('step_003', x)
        return
