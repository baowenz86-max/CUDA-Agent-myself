import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super().__init__()
        self.softplus = nn.Softplus()
        self.dim = dim

    def forward(self, x1, x2):
        x1 = cuda_extension.softplus_forward(x1, beta=1.0, threshold=20.0)
        x2 = cuda_extension.softplus_forward(x2, beta=1.0, threshold=20.0)
        x = cuda_extension.concat_forward(x1, x2, dim=self.dim)
        return x

    def debug_forward(self, x1, x2):
        x1 = cuda_extension.softplus_forward(x1, beta=1.0, threshold=20.0)
        yield 'step_001', x1
        x2 = cuda_extension.softplus_forward(x2, beta=1.0, threshold=20.0)
        yield 'step_002', x2
        x = cuda_extension.concat_forward(x1, x2, dim=self.dim)
        yield 'step_003', x
        yield 'step_004', x
