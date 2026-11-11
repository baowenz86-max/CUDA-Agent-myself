import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, leakiness):
        super().__init__()
        self.in_features = in_features
        self.negative_slope = leakiness

    def forward(self, x):
        return cuda_extension.custom_leaky_relu(x, self.negative_slope)

    def debug_forward(self, x):
        y = self.forward(x)
        yield 'step_001', y


batch_size = 1024
in_features = 128
height, width = 64, 64
leakiness = 0.01


def get_inputs():
    return [torch.randn(batch_size, in_features, height, width)]


def get_init_inputs():
    return [in_features, leakiness]
