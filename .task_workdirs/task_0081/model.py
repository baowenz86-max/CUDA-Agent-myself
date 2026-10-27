import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = F.silu(x)  # This operation is followed by permuting, so it might affect the shape
        x = torch.permute(x, (0, 2, 3, 1))  # Assuming x is a 4D tensor of shape (batch_size, channels, height, width)
        return x


batch_size = 512
in_features = 32
out_features = 64
width = 32
height = 160  # Increased from 32 to 160 to increase runtime


def get_inputs():
    return [torch.randn(batch_size, in_features, height, width)]


def get_init_inputs():
    return [in_features, out_features]