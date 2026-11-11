import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # No parameters or buffers are required; this matches the empty state_dict
        # of the reference Model.

    def forward(self, x, target):
        x = cuda_extension.custom_softmax(x, dim=1)
        x = cuda_extension.custom_cross_entropy(x, target)
        return x

    def debug_forward(self, *inputs):
        x, target = inputs
        x = cuda_extension.custom_softmax(x, dim=1)
        yield 'step_001', x
        x = cuda_extension.custom_cross_entropy(x, target)
        yield 'step_002', x
        yield 'step_003', x
