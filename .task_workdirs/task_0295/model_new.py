import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return cuda_extension.log_softmax_hardshrink(x)

    def debug_forward(self, x):
        y = cuda_extension.log_softmax_only(x)
        yield "step_001", y
        z = cuda_extension.hardshrink_only(y)
        yield "step_002", z
        yield "step_003", z
