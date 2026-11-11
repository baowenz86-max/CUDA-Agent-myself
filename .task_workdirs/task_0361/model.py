import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.hardswish = nn.Hardswish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hardswish(x)


batch_size = 16
in_features = 512
width = 64000  # Increased from 256 to increase computation


def get_inputs():
    x = torch.randn(batch_size, in_features, width)
    return [x]


def get_init_inputs():
    return [in_features]