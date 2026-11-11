import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = cuda_extension.layer_norm_forward(
            x,
            self.layer_norm.weight if self.elementwise_affine else None,
            self.layer_norm.bias if self.elementwise_affine else None,
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
            self.layer_norm.weight if self.elementwise_affine else None,
            self.layer_norm.bias if self.elementwise_affine else None,
            self.eps
        )
        yield 'step_001', normalized
        mean = cuda_extension.mean_dim_forward(
            normalized,
            1
        )
        yield 'step_002', mean
        yield 'step_003', mean
