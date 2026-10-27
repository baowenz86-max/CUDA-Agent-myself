import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.elu_sigmoid_forward(x)
        return x

    def debug_forward(self, x):
        elu_out = cuda_extension.elu_forward(x)
        yield "step_001", elu_out
        sigmoid_out = cuda_extension.sigmoid_forward(elu_out)
        yield "step_002", sigmoid_out
        yield "step_003", sigmoid_out


batch_size = 512
channels = 64
height = 128
width = 128


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return []
