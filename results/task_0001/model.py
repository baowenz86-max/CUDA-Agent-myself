import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, leakyrelu_negative_slope, pow_exponent, mish_alpha, relu_threshold, abs_max):
        super().__init__()
        self.leakyrelu = nn.LeakyReLU(negative_slope=leakyrelu_negative_slope)
        self.pow_exponent = pow_exponent
        self.mish_alpha = mish_alpha
        self.relu_threshold = relu_threshold
        self.abs_max = abs_max

    def forward(self, x):
        x = self.leakyrelu(x)
        x = torch.pow(x, self.pow_exponent)
        x = torch.abs(x)
        x = F.mish(x, self.mish_alpha)
        x = F.relu(x, self.relu_threshold)
        x = torch.clamp(x, min=0, max=self.abs_max)
        return x


batch_size = 512
in_channels = 64
height, width = 100, 100  # Increased spatial dimensions to increase computation


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [0.1, 2, 1, 0.2, 100]