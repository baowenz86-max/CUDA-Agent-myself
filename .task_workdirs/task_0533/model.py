import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.max(x, dim=1)[0]  # torch.max
        x = nn.Softmax(dim=1)(x)  # nn.Softmax
        return x


def get_inputs():
    x = torch.randn(64, 8192, 2048)  # Increased dimensions to increase runtime
    return [x]


def get_init_inputs():
    return []