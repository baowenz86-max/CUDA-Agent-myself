import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, dim, max_value):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.dim = dim
        self.max_value = max_value

    def forward(self, x):
        x = self.batch_norm(x)
        return torch.max(x, self.dim)[0] * self.max_value


# Define the parameters based on the model's needs
num_features = 64
dim = 1
max_value = 0.5


def get_inputs():
    # Create a random tensor with the specified input size
    batch_size = 16
    in_channels = num_features
    height, width = 1024, 128  # Increased spatial dimensions to increase runtime
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    # Return the parameters for the model's initialization
    return [num_features, dim, max_value]