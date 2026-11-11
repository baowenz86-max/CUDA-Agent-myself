import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape1, input_shape2, target_shape):
        super().__init__()
        self.input_shape1 = input_shape1
        self.input_shape2 = input_shape2
        self.target_shape = target_shape

    def forward(self, x1, x2, target):
        x = cuda_extension.cat_forward(x1, x2, dim=1)
        loss = cuda_extension.mse_loss_forward(x, target)
        return loss

    def debug_forward(self, x1, x2, target):
        x = cuda_extension.cat_forward(x1, x2, dim=1)
        yield ('step_001', x)
        loss = cuda_extension.mse_loss_forward(x, target)
        yield ('step_002', loss)
        yield ('step_003', loss)
