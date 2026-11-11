import torch
import torch.nn as nn
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, tensor_shape, mean, std):
        super().__init__()
        self.tensor = nn.Parameter(torch.normal(mean=mean, std=std, size=tensor_shape))

    def forward(self, x):
        x = cuda_extension.custom_max_broadcast(x, self.tensor, -1)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_max_broadcast(x, self.tensor, -1)
        yield ('step_001', x)
        yield ('step_002', x)
