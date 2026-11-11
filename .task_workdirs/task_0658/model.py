import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, zero_pad_amount, cat_dim):
        super().__init__()
        self.zero_pad = nn.ZeroPad2d(zero_pad_amount)
        self.cat_dim = cat_dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.zero_pad(x1)
        x2 = self.zero_pad(x2)
        return torch.cat((x1, x2), self.cat_dim)


batch_size = 512
channels = 3
height = 256  # Increased from 32 to 256 to increase computational load
width = 256   # Increased from 32 to 256 to increase computational load
zero_pad_amount = (1, 1, 1, 1)  # (left, right, top, bottom)
cat_dim = 1


def get_inputs():
    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    return [x1, x2]


def get_init_inputs():
    return [zero_pad_amount, cat_dim]