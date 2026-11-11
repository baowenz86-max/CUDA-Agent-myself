import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.channel_shuffle_forward(x, self.groups)

    def debug_forward(self, *inputs):
        x = inputs[0]
        out = cuda_extension.channel_shuffle_forward(x, self.groups)
        yield 'step_001', out
