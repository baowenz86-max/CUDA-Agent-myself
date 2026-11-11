import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, reduction_factor):
        super().__init__()
        self.reduction_factor = reduction_factor

    def forward(self, x):
        return cuda_extension.elu_abs_forward(x)

    def debug_forward(self, x):
        intermediate = cuda_extension.elu_forward(x)
        output = cuda_extension.abs_forward(intermediate)
        yield 'step_001', intermediate
        yield 'step_002', output
        yield 'step_003', output
