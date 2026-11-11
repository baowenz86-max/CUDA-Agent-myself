import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        x1 = cuda_extension.custom_softplus(x1)
        x2 = cuda_extension.custom_softplus(x2)
        x = cuda_extension.custom_cat(x1, x2, self.dim)
        return x

    def debug_forward(self, x1, x2):
        x1 = cuda_extension.custom_softplus(x1)
        yield "step_001", x1
        x2 = cuda_extension.custom_softplus(x2)
        yield "step_002", x2
        x = cuda_extension.custom_cat(x1, x2, self.dim)
        yield "step_003", x
        yield "step_004", x
