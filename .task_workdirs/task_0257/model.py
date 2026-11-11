import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, size, scale, align_corners):
        super().__init__()
        self.size = size
        self.scale = scale
        self.align_corners = align_corners

    def forward(self, theta):
        grid = torch.affine_grid_generator(theta, self.size, align_corners=self.align_corners)
        grid = grid * self.scale
        return grid


batch_size = 8
num_batches = 512  # Increased to ensure at least one dimension >=512
height = 256  # Increased from 128 to increase computation
width = 256   # Increased from 128 to increase computation
scale = 0.5
align_corners = True


def get_inputs():
    theta = torch.randn(num_batches, 2, 3)  # Shape remains (512, 2, 3)
    return [theta]


def get_init_inputs():
    return [[num_batches, 2, height, width], scale, align_corners]