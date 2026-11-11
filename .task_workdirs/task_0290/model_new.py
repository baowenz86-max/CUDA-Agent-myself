import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, n_channels, tril_offset):
        super().__init__()
        self.tril_offset = tril_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.tril_forward(x, self.tril_offset)

    def debug_forward(self, x: torch.Tensor):
        out = cuda_extension.tril_forward(x, self.tril_offset)
        yield 'step_001', out
