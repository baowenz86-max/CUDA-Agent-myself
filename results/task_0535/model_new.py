import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_values, _ = cuda_extension.cummax_forward(x, dim=0)
        return max_values

    def debug_forward(self, *inputs):
        x = inputs[0]
        values, indices = cuda_extension.cummax_forward(x, dim=0)
        yield 'step_001', (values, indices)
        yield 'step_002', values
