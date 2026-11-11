import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.randn(num_features))
        self.bias = nn.Parameter(torch.randn(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _, _ = torch.native_batch_norm(
            x, self.weight, self.bias, self.running_mean, self.running_var, self.training, self.momentum, self.eps
        )
        return x


batch_size = 1024  # Increased from 512 to increase runtime
num_features = 64
height, width = 32, 32


def get_inputs():
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    return [num_features]