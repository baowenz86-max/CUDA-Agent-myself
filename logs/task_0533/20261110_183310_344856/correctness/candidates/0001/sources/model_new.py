import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.max_reduce_forward(x, dim=1)
        x = cuda_extension.softmax_forward(x, dim=1)
        return x

    def debug_forward(self, x: torch.Tensor):
        max_out = cuda_extension.max_reduce_forward(x, dim=1)
        yield "step_001", max_out

        softmax_out = cuda_extension.softmax_forward(max_out, dim=1)
        yield "step_002", softmax_out

        yield "step_003", softmax_out
