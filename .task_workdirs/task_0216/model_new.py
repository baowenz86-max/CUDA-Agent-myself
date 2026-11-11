import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.gelu_forward(x)
        x = cuda_extension.hardswish_forward(x)
        return x

    def debug_forward(self, *inputs):
        """Yield checkpoint values in the same order as the reference trace."""
        x = inputs[0]
        x = cuda_extension.gelu_forward(x)
        yield "step_001", x
        x = cuda_extension.hardswish_forward(x)
        yield "step_002", x
        yield "step_003", x
