import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, _ = torch.max(x, dim=1, keepdim=True)  # Using torch.max
        return F.binary_cross_entropy_with_logits(x, torch.randn_like(x))  # Using F.binary_cross_entropy_with_logits


batch_size, height, width = 256, 512, 256  # Increased batch size to 256 to increase runtime


def get_inputs():
    return [torch.randn(batch_size, 1, height, width)]  # One channel as per torch.max operation


def get_init_inputs():
    return []  # No additional initialisation parameters required