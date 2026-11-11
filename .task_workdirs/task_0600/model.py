import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, num_filters, kernel_size, bias_shape):
        super(Model, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(num_features, num_filters, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.sigmoid(x)
        x = self.conv(x)
        x = x + self.bias
        return x


batch_size = 128
num_features = 3
num_filters = 16
height, width = 512, 64  # Increased width from 32 to 64 to increase runtime
kernel_size = (3, 3)
bias_shape = (num_filters, 1, 1)


def get_inputs():
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    return [num_features, num_filters, kernel_size, bias_shape]