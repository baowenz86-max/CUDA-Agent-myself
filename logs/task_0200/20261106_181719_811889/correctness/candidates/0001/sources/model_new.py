import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, x):
        return cuda_extension.hardshrink_elu_forward(x)

    def debug_forward(self, x):
        hardshrink_out = cuda_extension.hardshrink_forward(x)
        yield "step_001", hardshrink_out

        elu_out = cuda_extension.elu_forward(hardshrink_out)
        yield "step_002", elu_out

        yield "step_003", elu_out
