import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = cuda_extension.layer_norm_forward(
            x,
            self.weight,
            self.bias,
            self.eps
        )
        mean = cuda_extension.mean_dim_forward(
            normalized,
            1
        )
        return mean

    def debug_forward(self, *inputs):
        x = inputs[0]
        normalized = cuda_extension.layer_norm_forward(
            x,
            self.weight,
            self.bias,
            self.eps
        )
        yield 'step_001', normalized
        mean = cuda_extension.mean_dim_forward(
            normalized,
            1
        )
        yield 'step_002', mean
        yield 'step_003', mean
