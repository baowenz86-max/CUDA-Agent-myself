import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.relu(x)
        x = torch.trunc(x)
        return x

    def debug_forward(self, x):
        relu_out = torch.relu(x)
        yield 'step_001', relu_out
        trunc_out = torch.trunc(relu_out)
        yield 'step_002', trunc_out
        yield 'step_003', trunc_out
