import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.celu = nn.CELU()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.linear_celu(x, self.linear.weight, self.linear.bias)

    def debug_forward(self, *inputs):
        x = inputs[0]
        y = cuda_extension.linear_celu(x, self.linear.weight, self.linear.bias)
        yield 'step_001', y
