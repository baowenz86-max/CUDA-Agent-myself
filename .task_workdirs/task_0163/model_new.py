import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return cuda_extension.softplus_forward(x)

    def debug_forward(self, x):
        out = cuda_extension.softplus_forward(x)
        yield 'step_001', out
