import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.scalar_tensor = torch.scalar_tensor(1.0)

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = x * self.scalar_tensor
        return x


batch_size = 512
num_features = 3


def get_inputs():
    x = torch.randn(batch_size, num_features, 1024, 1024)  # Increased spatial dimensions
    return [x]


def get_init_inputs():
    return [num_features]