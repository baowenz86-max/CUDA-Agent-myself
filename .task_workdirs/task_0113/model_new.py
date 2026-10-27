import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return cuda_extension.argmax_dim1_forward(x)

    def debug_forward(self, x):
        out = cuda_extension.argmax_dim1_forward(x)
        yield "step_001", out


batch_size = 16
channels = 3
height = 4096
width = 1024


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
