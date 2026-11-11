import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return cuda_extension.custom_l1_loss_mean(input, target)

    def debug_forward(self, input, target):
        result = cuda_extension.custom_l1_loss_mean(input, target)
        yield 'step_001', result
