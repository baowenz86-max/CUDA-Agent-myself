import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, x):
        x = torch.max(x, dim=1)[0]  # Max pooling along channel dimension
        x = torch.exp(x)  # Apply exponential function
        return x


num_features = 64
input_shape = (32, num_features, 512, 512)  # Added batch dimension and increased size


def get_inputs():
    return [torch.randn(*input_shape)]


def get_init_inputs():
    return [num_features]