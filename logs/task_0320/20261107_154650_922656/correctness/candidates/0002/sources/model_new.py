import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.hardtanh_forward(x)
        x = cuda_extension.leaky_relu_forward(x)
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.hardtanh_forward(x)
        yield 'step_001', x1
        x2 = cuda_extension.leaky_relu_forward(x1)
        yield 'step_002', x2
        yield 'step_003', x2
