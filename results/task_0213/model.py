import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, threshold_value, out_features):
        super().__init__()
        self.threshold = nn.Threshold(threshold_value, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.threshold(x)


batch_size = 64  # Increased from 16 to 64 to increase input size
out_features = 64
threshold_value = 0.5
width = 256
height = 512  # Maintained at 512 as per original modification


def get_inputs():
    x = torch.randn(batch_size, out_features, height, width)
    return [x]


def get_init_inputs():
    return [threshold_value, out_features]