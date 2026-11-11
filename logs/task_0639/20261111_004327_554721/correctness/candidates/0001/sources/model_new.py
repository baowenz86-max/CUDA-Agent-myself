import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, num_channels, eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x_max, _ = cuda_extension.custom_max_dim1_keepdim(x)
        x = cuda_extension.custom_layer_norm(x_max, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps)
        return x

    def debug_forward(self, x):
        x_max, indices = cuda_extension.custom_max_dim1_keepdim(x)
        yield 'step_001', (x_max, indices)
        x_norm = cuda_extension.custom_layer_norm(x_max, self.layer_norm.weight, self.layer_norm.bias, self.layer_norm.eps)
        yield 'step_002', x_norm
        yield 'step_003', x_norm
