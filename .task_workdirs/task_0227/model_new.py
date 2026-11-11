import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, x):
        return cuda_extension.hardshrink_forward(x)

    def debug_forward(self, x):
        y = cuda_extension.hardshrink_forward(x)
        yield "step_001", y
