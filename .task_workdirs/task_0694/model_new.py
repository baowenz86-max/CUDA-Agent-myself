import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_axes):
        super().__init__()
        self.num_axes = num_axes

    def forward(self, x):
        x = cuda_extension.custom_sum_and_sigmoid(x, self.num_axes)
        return x

    def debug_forward(self, x):
        sum_result = cuda_extension.custom_sum_only(x, self.num_axes)
        yield ('step_001', sum_result)
        sigmoid_result = cuda_extension.custom_sigmoid_only(sum_result)
        yield ('step_002', sigmoid_result)
        yield ('step_003', sigmoid_result)
