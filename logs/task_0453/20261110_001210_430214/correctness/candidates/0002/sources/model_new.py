import torch
import torch.nn as nn
import torch.nn.functional as F
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, y):
        result = cuda_extension.custom_equal(x, y)
        result = cuda_extension.custom_l1_loss(x, y)
        return result

    def debug_forward(self, x, y):
        yield 'step_001', cuda_extension.custom_equal(x, y)
        result = cuda_extension.custom_l1_loss(x, y)
        yield 'step_002', result
        yield 'step_003', result
