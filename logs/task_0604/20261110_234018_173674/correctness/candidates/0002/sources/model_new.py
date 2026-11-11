import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.abs_relu6_forward(x)
        return x

    def debug_forward(self, x):
        abs_x = cuda_extension.abs_forward(x)
        yield 'step_001', abs_x
        relu6_x = cuda_extension.relu6_forward(abs_x)
        yield 'step_002', relu6_x
        yield 'step_003', relu6_x
