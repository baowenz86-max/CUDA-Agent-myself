import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, pool_size):
        super().__init__()
        self.elu = nn.ELU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=pool_size)

    def forward(self, x):
        x = self.elu(x)
        x = self.max_pool2d(x)
        return x


in_features = 512
pool_size = (2, 2)


def get_inputs():
    return [torch.randn(16, in_features, 256, 256)]  # Increased spatial dimensions to 256x256


def get_init_inputs():
    return [in_features, pool_size]