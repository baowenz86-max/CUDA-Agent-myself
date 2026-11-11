import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps, momentum):
        super().__init__()
        self.bn2d = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = torch.mean(x, dim=0, keepdim=True)
        x = self.bn2d(x)
        return x


batch_size = 64
num_features = 3
height, width = 2048, 1024  # Increased spatial dimensions to increase computation
eps = 1e-5
momentum = 0.1


def get_inputs():
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    return [num_features, eps, momentum]