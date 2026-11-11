import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.celu = nn.CELU()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.celu(self.linear(x))


batch_size = 8192
in_features = 2048
out_features = 1024


def get_inputs():
    x = torch.randn(batch_size, in_features)
    return [x]


def get_init_inputs():
    return [in_features, out_features]