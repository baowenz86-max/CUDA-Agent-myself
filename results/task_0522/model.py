import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x):
        x1 = torch.equal(x, x)  # You cannot use x1 to hold the result of torch.equal, you can remove this line
        x = self.batch_norm(x)
        return x


batch_size = 512  # Changed from 64 to 512 to ensure at least one dimension >=512
num_features = 3
height, width = 128, 128  # Increased spatial dimensions to increase computation


def get_inputs():
    return [torch.randn(batch_size, num_features, height, width)]


def get_init_inputs():
    return [num_features]