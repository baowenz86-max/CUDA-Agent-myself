import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = cuda_extension.custom_softmax(x, self.dim)
        x = cuda_extension.custom_unsqueeze(x, self.dim)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_softmax(x, self.dim)
        yield 'step_001', x
        x = cuda_extension.custom_unsqueeze(x, self.dim)
        yield 'step_002', x
        yield 'step_003', x
