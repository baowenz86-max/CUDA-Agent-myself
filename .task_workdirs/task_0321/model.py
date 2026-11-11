import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # No need for any initializations as we are not using any trainable parameters

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(x)
        x = F.cosine_similarity(x, y)
        return x


batch_size = 512


def get_inputs():
    x = torch.randn(batch_size, 3, 256, 256)
    y = torch.randn(batch_size, 3, 256, 256)
    return [x, y]


def get_init_inputs():
    return []  # No parameters to initialize