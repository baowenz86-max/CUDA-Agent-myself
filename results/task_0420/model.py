import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        equal_result = torch.equal(x, y)
        loss = F.smooth_l1_loss(x, y)
        return equal_result, loss


batch_size = 512
height = 512
width = 512


def get_inputs():
    x = torch.randn(batch_size, height, width)
    y = torch.randn(batch_size, height, width)
    return [x, y]


def get_init_inputs():
    return []