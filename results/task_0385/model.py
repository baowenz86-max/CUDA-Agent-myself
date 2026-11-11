import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, normalized_shape, eps, elementwise_affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = torch.clamp(x, min=-1.0, max=1.0)  # torch.clamp
        x = self.layer_norm(x)  # nn.LayerNorm
        return x


# Define the input shape and test data
batch_size, seq_len, feat_dim = 512, 512, 768
normalized_shape = (feat_dim,)
eps = 1e-5


# Generate random inputs
def get_inputs():
    x = torch.randn(batch_size, seq_len, feat_dim)
    return [x]


# Get the required initialization parameters
def get_init_inputs():
    return [normalized_shape, eps]