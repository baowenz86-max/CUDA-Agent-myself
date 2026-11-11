import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        equal_result = cuda_extension.custom_equal(x, y)
        loss = cuda_extension.custom_smooth_l1_loss(x, y)
        return equal_result, loss

    def debug_forward(self, x, y):
        equal_result = cuda_extension.custom_equal(x, y)
        yield 'step_001', equal_result
        loss = cuda_extension.custom_smooth_l1_loss(x, y)
        yield 'step_002', loss
        yield 'step_003', (equal_result, loss)
