import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, leaky_relu_negative_slope, batch_norm_num_features):
        super().__init__()
        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.batch_norm = nn.BatchNorm3d(num_features=batch_norm_num_features)

    def forward(self, x):
        x = self.tanh(x)
        x = self.leaky_relu(x)
        x = self.batch_norm(x)
        return x


batch_size = 8
in_channels = 3
out_channels = 64
leaky_relu_negative_slope = 0.01
height = 256  # Reduced from 512
width = 128   # Reduced from 256
depth = 64    # Reduced from 128
batch_norm_num_features = in_channels


def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, leaky_relu_negative_slope, batch_norm_num_features]