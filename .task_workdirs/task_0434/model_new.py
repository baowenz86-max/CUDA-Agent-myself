import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.custom_abs(x)
        x = cuda_extension.custom_cumsum(x, 0)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_abs(x)
        yield 'step_001', x
        x = cuda_extension.custom_cumsum(x, 0)
        yield 'step_002', x
        yield 'step_003', x
