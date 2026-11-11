import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return cuda_extension.softmin_dim1(x)

    def debug_forward(self, x):
        y = cuda_extension.softmin_dim1(x)
        yield 'step_001', y
