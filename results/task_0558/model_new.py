import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.hardsigmoid_forward(x.cuda())

    def debug_forward(self, x: torch.Tensor):
        out = cuda_extension.hardsigmoid_forward(x.cuda())
        yield 'step_001', out
