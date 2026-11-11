import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.custom_softmax(x, 1)
        x = cuda_extension.custom_softmax(x, 1)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_softmax(x, 1)
        yield 'step_001', x
        x = cuda_extension.custom_softmax(x, 1)
        yield 'step_002', x
        yield 'step_003', x
