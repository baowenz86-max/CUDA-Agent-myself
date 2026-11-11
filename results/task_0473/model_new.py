import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.hardtanh_forward(x, -1.0, 1.0)

    def debug_forward(self, *inputs):
        x = inputs[0]
        y = cuda_extension.hardtanh_forward(x, -1.0, 1.0)
        yield 'step_001', y
