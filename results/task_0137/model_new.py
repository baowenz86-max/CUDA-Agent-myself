import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = cuda_extension.mean_dim1(x)
        loss = cuda_extension.smooth_l1_loss(x, y)
        return loss

    def debug_forward(self, x, y):
        x_mean = cuda_extension.mean_dim1(x)
        yield 'step_001', x_mean
        loss = cuda_extension.smooth_l1_loss(x_mean, y)
        yield 'step_002', loss
        yield 'step_003', loss
