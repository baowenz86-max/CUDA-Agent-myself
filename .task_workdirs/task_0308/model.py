import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, normalized_shape, eps, elementwise_affine):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = self.layernorm(x)
        x = torch.cumprod(x, dim=1)
        return x


batch_size = 16
normalized_shape = torch.Size([512, 2048])  # Increased first dimension from 16 to 512
eps = 1e-5
elementwise_affine = True


def get_inputs():
    x = torch.randn(batch_size, *normalized_shape)
    return [x]


def get_init_inputs():
    return [normalized_shape, eps, elementwise_affine]