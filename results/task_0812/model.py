import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, scale_factor, mode, rotation_k):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.rotation_k = rotation_k

    def forward(self, x):
        x = self.upsample(x)
        x = torch.rot90(x, k=self.rotation_k, dims=(2, 3))
        return x


batch_size = 16
height, width = 512, 128  # Increased height to 512 to ensure at least one dimension >=512
scale_factor = 2.0
mode = "bilinear"
rotation_k = 1


def get_inputs():
    return [torch.randn(batch_size, 64, height, width)]  # Increased channels to 64 to increase computation


def get_init_inputs():
    return [scale_factor, mode, rotation_k]