import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, normalized_shape, eps, elementwise_affine):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = self.layer_norm(x)
        return x


batch_size = 1024  # Increased from 16 to 1024 to increase computation
seq_len = 512
feature_dim = 64


def get_inputs():
    x = torch.randn(batch_size, seq_len, feature_dim)
    return [x]


def get_init_inputs():
    return [normalized_shape, eps, elementwise_affine]


normalized_shape = (feature_dim,)
eps = 1e-5
elementwise_affine = True