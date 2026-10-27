import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.relu_clamp_forward(x)
        return x

    def debug_forward(self, x):
        relu_result = cuda_extension.relu_clamp_forward(x)
        yield ("step_001", relu_result)
        clamp_result = cuda_extension.relu_clamp_forward(x)
        yield ("step_002", clamp_result)
        result = cuda_extension.relu_clamp_forward(x)
        yield ("step_003", result)


batch_size = 1792
height = 512
width = 256


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return []
