import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps, affine=True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, eps, affine=affine)

    def forward(self, x):
        x = self.instance_norm(x)
        x = torch.abs(x)  # Apply torch.abs operation
        return x


# Define some test parameters
batch_size = 32
channels = 3
height = 1024  # Increased from 512 to 1024
width = 1024   # Increased from 64 to 1024
num_features = channels
eps = 1e-5


def get_inputs():
    # A batch of random input data with size [batch_size, channels, height, width]
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    # Return the parameters needed for Model initialization
    return [num_features, eps]