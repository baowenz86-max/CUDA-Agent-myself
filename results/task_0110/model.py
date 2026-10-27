import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps, momentum):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        self.clamp_lower_bound = 0.0
        self.clamp_upper_bound = 1.0

    def forward(self, x):
        x = torch.clamp(x, min=self.clamp_lower_bound, max=self.clamp_upper_bound)
        x = self.batch_norm(x)
        return x


batch_size = 128
num_features = 3
height, width = 1024, 32  # Increased height to 1024 to increase runtime
momentum = 0.1
eps = 1e-5


def get_inputs():
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    return [num_features, eps, momentum]