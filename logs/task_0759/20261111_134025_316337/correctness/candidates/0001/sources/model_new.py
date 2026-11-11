import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        softmax_out = cuda_extension.softmax(x, dim=1)
        hardswish_out = cuda_extension.hardswish(softmax_out)
        return hardswish_out

    def debug_forward(self, x):
        softmax_out = cuda_extension.softmax(x, dim=1)
        yield 'step_001', softmax_out
        hardswish_out = cuda_extension.hardswish(softmax_out)
        yield 'step_002', hardswish_out
        yield 'step_003', hardswish_out
