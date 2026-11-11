import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

    def forward(self, x: torch.Tensor):
        max_value = cuda_extension.reduce_max(x)
        std_value = cuda_extension.reduce_std(x)
        return max_value, std_value

    def debug_forward(self, x: torch.Tensor):
        max_value = cuda_extension.reduce_max(x)
        yield "step_001", max_value
        std_value = cuda_extension.reduce_std(x)
        yield "step_002", std_value
        result = (max_value, std_value)
        yield "step_003", result
