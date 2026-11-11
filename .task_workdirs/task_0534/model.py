import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        max_values, _ = torch.max(x, dim=1, keepdim=True)
        similarity = F.cosine_similarity(x, max_values, dim=1)
        return similarity


batch_size = 64
width = 128
height = 8192  # Increased from 1024 to increase computation


def get_inputs():
    x = torch.randn(batch_size, height, width)
    return [x]


def get_init_inputs():
    return []  # No inputs needed for this model initialization