import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, normalized_shape, eps, elementwise_affine):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = cuda_extension.layer_norm_forward(
            x,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.layer_norm.normalized_shape[0],
            self.layer_norm.eps
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.layer_norm_forward(
            x,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.layer_norm.normalized_shape[0],
            self.layer_norm.eps
        )
        yield 'step_001', x
        yield 'step_002', x
