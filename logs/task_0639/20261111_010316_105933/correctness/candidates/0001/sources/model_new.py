import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x, _ = cuda_extension.max_reduce(x, dim=1, keepdim=True)
        x = cuda_extension.layer_norm_forward(
            x, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps
        )
        return x

    def debug_forward(self, x):
        max_values, max_indices = cuda_extension.max_reduce(x, dim=1, keepdim=True)
        yield 'step_001', (max_values, max_indices)
        normalized = cuda_extension.layer_norm_forward(
            max_values, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps
        )
        yield 'step_002', normalized
        yield 'step_003', normalized
