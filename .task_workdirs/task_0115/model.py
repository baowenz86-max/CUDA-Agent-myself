import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, upscale_factor, norm_p, norm_dim):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.norm_p = norm_p
        self.norm_dim = norm_dim

    def forward(self, x):
        x = torch.norm(x, p=self.norm_p, dim=self.norm_dim)
        x = self.pixel_shuffle(x.unsqueeze(-1).unsqueeze(-1))
        return x


batch_size = 512
channels = 32
height, width = 1024, 1024  # Increased spatial dimensions to increase computation
upscale_factor = 2
norm_p = 2
norm_dim = 1


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [upscale_factor, norm_p, norm_dim]