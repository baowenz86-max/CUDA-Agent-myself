import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x = F.softmax(x, dim=1)
        loss = nn.MSELoss()(x, y)
        return loss


batch_size = 512
classes = 10
height = 128  # Increased from 32 to 128
width = 128   # Increased from 32 to 128


def get_inputs():
    x = torch.randn(batch_size, classes, height, width)
    y = torch.randn(batch_size, classes, height, width)
    return [x, y]


def get_init_inputs():
    return []