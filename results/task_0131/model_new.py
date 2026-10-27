import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        x = cuda_extension.custom_relu(x)
        x = cuda_extension.custom_gelu(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_relu(x)
        yield ('step_001', x)
        x = cuda_extension.custom_gelu(x)
        yield ('step_002', x)
        yield ('step_003', x)
