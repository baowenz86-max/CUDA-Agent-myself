import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps):
        super().__init__()
        self.relu = nn.ReLU()
        self.instance_norm = nn.InstanceNorm3d(num_features, eps=eps)

    def forward(self, x):
        x = self.relu(x)
        x = self.instance_norm(x)
        return x


batch_size = 512
features_channels = 32
depth, height, width = 18, 18, 18


def get_inputs():
    # Assuming a batch of 3D features with shape [batch_size, features_channels, depth, height, width]
    return [torch.randn(batch_size, features_channels, depth, height, width)]


def get_init_inputs():
    # Number of features/channels
    num_features = 32
    # Small value to maintain numerical stability
    eps = 1e-5
    return [num_features, eps]