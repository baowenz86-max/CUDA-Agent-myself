import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.batch_norm2d = nn.BatchNorm2d(1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tanh(x)
        x = self.batch_norm2d(x)
        return x


def get_inputs():
    return [torch.randn(256, 1024, 20, 30)]


def get_init_inputs():
    return []