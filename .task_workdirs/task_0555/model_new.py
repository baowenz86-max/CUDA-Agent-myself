import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, normalize_shape, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x, _ = cuda_extension.custom_max_dim1(x)
        x = cuda_extension.custom_normalize(x, self.dim)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        max_result = cuda_extension.custom_max_dim1(x)
        yield ('step_001', max_result)
        x = max_result[0]
        normalized = cuda_extension.custom_normalize(x, self.dim)
        yield ('step_002', normalized)
        yield ('step_003', normalized)
        return


def get_inputs():
    return [torch.randn(128, 32, 1024, 256)]


def get_init_inputs():
    return [None, 2]
