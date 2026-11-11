import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, target):
        x = cuda_extension.custom_logical_or(x, y, 0.5)
        x = cuda_extension.custom_cast_float(x)
        loss = cuda_extension.custom_mse_loss_mean(x, target)
        return loss

    def debug_forward(self, *inputs):
        x, y, target = inputs
        x = cuda_extension.custom_logical_or(x, y, 0.5)
        yield 'step_001', x
        x = cuda_extension.custom_cast_float(x)
        yield 'step_002', x
        loss = cuda_extension.custom_mse_loss_mean(x, target)
        yield 'step_003', loss
        yield 'step_004', loss
