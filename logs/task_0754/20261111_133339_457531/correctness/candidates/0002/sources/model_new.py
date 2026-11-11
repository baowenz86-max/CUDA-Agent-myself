import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, num_features):
        super().__init__()
        self.in_channels = in_channels
        self.num_features = num_features

    def forward(self, x):
        # Call custom extension op that performs cat along dim=1 and maximum
        y = cuda_extension.cat_maximum_forward(x)
        return y

    def debug_forward(self, x):
        # Yield intermediate values exactly as reference checkpoints
        x1 = cuda_extension.cat_maximum_forward(x)  # step_001
        yield 'step_001', x1
        x2 = cuda_extension.cat_maximum_forward(x)  # step_002
        yield 'step_002', x2
        yield 'step_003', x2
