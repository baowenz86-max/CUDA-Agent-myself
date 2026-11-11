import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = cuda_extension.cat_forward(x1, x2, dim=1)
        x = cuda_extension.sigmoid_forward(x)
        return x

    def debug_forward(self, x1, x2):
        x = cuda_extension.cat_forward(x1, x2, dim=1)
        yield "step_001", x
        x = cuda_extension.sigmoid_forward(x)
        yield "step_002", x
        yield "step_003", x
