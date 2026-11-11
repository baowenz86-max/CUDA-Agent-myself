import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        x = torch.cos(x)
        x = self.identity(x)
        return x

    def debug_forward(self, x):
        x = torch.cos(x)
        yield 'step_001', x
        x = self.identity(x)
        yield 'step_002', x
        yield 'step_003', x
