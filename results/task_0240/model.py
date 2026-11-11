import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.binary_cross_entropy_with_logits(y_pred, y_true)


batch_size = 128  # Increased from 16
width = 1024  # Increased from 512
height = 1024  # Increased from 256


def get_inputs():
    y_pred = torch.randn(batch_size, 1, height, width)
    y_true = torch.randn(batch_size, 1, height, width)
    return [y_pred, y_true]


def get_init_inputs():
    return []