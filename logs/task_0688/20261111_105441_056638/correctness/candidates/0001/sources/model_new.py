import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_channels, target_channels):
        super().__init__()
        self.target_channels = target_channels

    def forward(self, x1, x2, target):
        x = cuda_extension.custom_cat(x1, x2, dim=1)
        loss = cuda_extension.custom_l1_loss(x, target)
        return loss

    def debug_forward(self, x1, x2, target):
        x = cuda_extension.custom_cat(x1, x2, dim=1)
        yield 'step_001', x
        loss = cuda_extension.custom_l1_loss(x, target)
        yield 'step_002', loss
        yield 'step_003', loss
