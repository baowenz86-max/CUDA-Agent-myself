import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        x_clone = cuda_extension.clone_forward(x)
        return x_clone

    def debug_forward(self, x):
        x_clone = cuda_extension.clone_forward(x)
        yield ('step_001', x_clone)
        yield ('step_002', x_clone)
