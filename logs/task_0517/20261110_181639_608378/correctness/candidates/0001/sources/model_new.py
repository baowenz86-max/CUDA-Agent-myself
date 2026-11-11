import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.circular_pad_forward(x, self.padding)

    def debug_forward(self, x: torch.Tensor):
        out = cuda_extension.circular_pad_forward(x, self.padding)
        yield ('step_001', out)
