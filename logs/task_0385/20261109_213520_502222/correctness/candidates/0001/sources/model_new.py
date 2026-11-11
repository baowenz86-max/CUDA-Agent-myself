import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, normalized_shape, eps, elementwise_affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = cuda_extension.clamp_forward(x, min_val=-1.0, max_val=1.0)
        x = cuda_extension.layer_norm_forward(
            x,
            self.layer_norm.weight,
            self.layer_norm.bias,
            eps=1e-5
        )
        return x

    def debug_forward(self, x):
        clamped = cuda_extension.clamp_forward(x, min_val=-1.0, max_val=1.0)
        normalized = cuda_extension.layer_norm_forward(
            clamped,
            self.layer_norm.weight,
            self.layer_norm.bias,
            eps=1e-5
        )
        yield 'step_001', clamped
        yield 'step_002', normalized
        yield 'step_003', normalized
