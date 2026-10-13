import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.sigmoid_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.sigmoid_forward(x)
        yield 'step_001', x
        yield 'step_002', x
