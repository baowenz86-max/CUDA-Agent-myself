import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = torch.mean(x, dim=1)
        loss = F.smooth_l1_loss(x, y)
        return loss


batch_size = 512
channel = 128
height, width = 128, 128  # Increased spatial dimensions to increase computation


def get_inputs():
    x = torch.randn(batch_size, channel, height, width)
    y = torch.randn(batch_size, height, width)
    return [x, y]


def get_init_inputs():
    return []