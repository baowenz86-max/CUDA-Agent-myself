import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Preserve the module structure if needed; no parameters here.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.hardtanh_forward(x)

    def debug_forward(self, x: torch.Tensor):
        output = cuda_extension.hardtanh_forward(x)
        yield 'step_001', output
