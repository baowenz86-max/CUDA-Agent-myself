import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features=in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.custom_transpose(x, 1, 2)
        x = cuda_extension.custom_selu(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.custom_transpose(x, 1, 2)
        yield 'step_001', x
        x = cuda_extension.custom_selu(x)
        yield 'step_002', x
        yield 'step_003', x
