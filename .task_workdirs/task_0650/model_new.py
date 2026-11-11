import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape, index_shape, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, index):
        mask = cuda_extension.isnan_forward(x)
        x_gathered = cuda_extension.gather_forward(x, self.dim, index)
        return mask, x_gathered

    def debug_forward(self, x, index):
        mask = cuda_extension.isnan_forward(x)
        yield 'step_001', mask
        x_gathered = cuda_extension.gather_forward(x, self.dim, index)
        yield 'step_002', x_gathered
        yield 'step_003', (mask, x_gathered)
