import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.custom_sum_dim1_keepdim(x)
        x = cuda_extension.custom_nan_to_num(x)
        return x

    def debug_forward(self, *inputs):
        if len(inputs) != 1:
            raise ValueError("Expected exactly 1 input tensor")
        x = inputs[0]
        yield "step_001", cuda_extension.custom_sum_dim1_keepdim(x)
        x = cuda_extension.custom_nan_to_num(x)
        yield "step_002", x
        yield "step_003", x
