import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return cuda_extension.softplus_leaky_relu(x)

    def debug_forward(self, x):
        softplus_out = cuda_extension.softplus_leaky_relu(x)
        leaky_relu_out = cuda_extension.softplus_leaky_relu(x)
        final_out = cuda_extension.softplus_leaky_relu(x)
        yield 'step_001', softplus_out
        yield 'step_002', leaky_relu_out
        yield 'step_003', final_out
