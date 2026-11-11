import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, size, scale, align_corners):
        super().__init__()
        self.size = size
        self.scale = scale
        self.align_corners = align_corners

    def forward(self, theta):
        grid = cuda_extension.affine_grid_scaled(
            theta,
            self.size[2],
            self.size[3],
            self.align_corners,
            self.scale,
        )
        return grid

    def debug_forward(self, theta):
        grid = cuda_extension.affine_grid_scaled(
            theta,
            self.size[2],
            self.size[3],
            False,  # reference step_001 uses align_corners=self.align_corners which is True, but we may need to debug both steps
            self.scale,
        )
        yield "step_001", grid
        grid_scaled = grid * self.scale
        yield "step_002", grid_scaled
        yield "step_003", grid_scaled
