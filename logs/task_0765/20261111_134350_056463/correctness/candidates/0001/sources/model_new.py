import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        stacked = cuda_extension.custom_stack(x1, x2)
        result = cuda_extension.custom_bmm_transpose(stacked)
        return result

    def debug_forward(self, *inputs):
        x1, x2 = inputs
        stacked = cuda_extension.custom_stack(x1, x2)
        yield 'step_001', stacked
        result = cuda_extension.custom_bmm_transpose(stacked)
        yield 'step_002', result
        yield 'step_003', result
