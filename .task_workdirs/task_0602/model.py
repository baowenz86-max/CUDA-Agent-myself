import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super().__init__()
        self.softplus = nn.Softplus()
        self.dim = dim

    def forward(self, x1, x2):
        x1 = self.softplus(x1)
        x2 = self.softplus(x2)
        x = torch.cat((x1, x2), dim=self.dim)
        return x


batch_size = 64
height, width = 1024, 512  # Increased spatial dimensions to increase computation
in_channels = 3
out_channels = 64
dim = 1


def get_init_inputs():
    return [in_channels, out_channels, dim]


def get_inputs():
    x1 = torch.randn(batch_size, in_channels, height, width)
    x2 = torch.randn(batch_size, in_channels, height, width)
    return [x1, x2]