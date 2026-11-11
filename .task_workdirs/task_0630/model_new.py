import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = cuda_extension.custom_abs(x)
        x = cuda_extension.custom_reduce_lastdim(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_abs(x)
        yield 'step_001', x
        x = cuda_extension.custom_reduce_lastdim(x)
        yield 'step_002', x
        yield 'step_003', x
