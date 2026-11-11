import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, divisor: int = 1):
        super().__init__()
        self.divisor = divisor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.abs_div_forward(x, self.divisor)

    def debug_forward(self, x: torch.Tensor):
        # Debug interface yields the same checkpoints as reference_steps
        abs_x = cuda_extension.abs_div_forward(x, 1)  # divisor=1 gives only abs
        yield 'step_001', abs_x
        div_x = cuda_extension.abs_div_forward(x, self.divisor)
        yield 'step_002', div_x
        yield 'step_003', div_x
