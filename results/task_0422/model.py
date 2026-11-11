import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        loss = nn.MSELoss()(max_val, y)
        return loss


def get_inputs():
    batch_size = 512
    x = torch.randn(batch_size, 16, 256, 256)  # Increased channels from 1 to 16
    y = torch.randn(batch_size, 1, 1, 1)
    return [x, y]


def get_init_inputs():
    return []