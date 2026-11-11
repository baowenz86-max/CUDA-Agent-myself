import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size_1, input_size_2, squeeze_dim):
        super().__init__()
        self.squeeze_dim = squeeze_dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = torch.squeeze(x1, dim=self.squeeze_dim)
        x2 = torch.squeeze(x2, dim=self.squeeze_dim)
        return torch.cat((x1, x2), dim=self.squeeze_dim)


batch_size = 512
input_size_1 = 8
input_size_2 = 8
height = 256  # Increased from 64
width = 256   # Increased from 64
squeeze_dim = 1


def get_inputs():
    x1 = torch.randn(batch_size, input_size_1, height, width)
    x2 = torch.randn(batch_size, input_size_2, height, width)
    return [x1, x2]


def get_init_inputs():
    return [input_size_1, input_size_2, squeeze_dim]