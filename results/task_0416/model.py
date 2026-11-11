import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features=in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        x = F.selu(x)
        return x


batch_size = 128  # Increased from 16 to 128 to increase runtime
in_features = 32
height = 512
width = 256


def get_inputs():
    x = torch.randn(batch_size, in_features, height, width)
    return [x]


def get_init_inputs():
    return [in_features]