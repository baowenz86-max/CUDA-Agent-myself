import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, batchnorm_num_features, pooling_kernel_size):
        super().__init__()
        self.batchnorm2d = nn.BatchNorm2d(batchnorm_num_features)
        self.pooling = nn.AvgPool2d(pooling_kernel_size)

    def forward(self, x):
        x = self.batchnorm2d(x)
        x = torch.pow(x, 2)
        x = self.pooling(x)
        return x


batch_size = 512
num_features = 32
height, width = 64, 64  # Increased from 32x32 to 64x64
kernel_size = 2


def get_inputs():
    x = torch.randn(batch_size, num_features, height, width)
    return [x]


def get_init_inputs():
    return [num_features, kernel_size]