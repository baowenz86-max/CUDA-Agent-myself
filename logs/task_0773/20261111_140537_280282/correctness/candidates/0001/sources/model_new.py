import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dimension):
        super(ModelNew, self).__init__()
        self.dimension = dimension

    def forward(self, x, y):
        stacked_tensors = cuda_extension.stack_dim0(x, y, self.dimension)
        result = cuda_extension.all_dim0(stacked_tensors, self.dimension)
        return result

    def debug_forward(self, x, y):
        stacked_tensors = cuda_extension.stack_dim0(x, y, self.dimension)
        yield 'step_001', stacked_tensors
        result = cuda_extension.all_dim0(stacked_tensors, self.dimension)
        yield 'step_002', result
        yield 'step_003', result
