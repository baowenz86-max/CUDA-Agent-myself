import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.relu_transform(x)
        x = cuda_extension.smooth_l1_loss_mean_zero(x)
        return x

    def debug_forward(self, x):
        y = cuda_extension.relu_transform(x)
        yield ('step_001', y)
        loss = cuda_extension.smooth_l1_loss_mean_zero(y)
        yield ('step_002', loss)
        yield ('step_003', loss)
