import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.cuda_extension.softmin_dim1(x)

    def debug_forward(self, x):
        y = torch.ops.cuda_extension.softmin_dim1(x)
        yield 'step_001', y
