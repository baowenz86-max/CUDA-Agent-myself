import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.hardswish_forward(x)

    def debug_forward(self, x: torch.Tensor):
        output = cuda_extension.hardswish_forward(x)
        yield 'step_001', output


def get_inputs():
    x = torch.randn(16, 512, 64000)
    return [x]


def get_init_inputs():
    return [512]
