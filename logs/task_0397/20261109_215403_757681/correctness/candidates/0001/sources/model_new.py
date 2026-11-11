import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, lambd: float = 0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import cuda_extension
        return cuda_extension.softshrink_forward(x, self.lambd)

    def debug_forward(self, *inputs):
        import cuda_extension
        x = inputs[0]
        out = cuda_extension.softshrink_forward(x, self.lambd)
        yield 'step_001', out
