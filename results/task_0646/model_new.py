import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.sum_dim1(x)
        x = cuda_extension.log_softmax_dim0(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        step_001 = cuda_extension.sum_dim1(x)
        yield "step_001", step_001
        step_002 = cuda_extension.log_softmax_dim0(step_001)
        yield "step_002", step_002
        step_003 = step_002
        yield "step_003", step_003
