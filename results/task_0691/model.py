import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss = F.mse_loss(x, y)
        sum_loss = torch.sum(loss)
        return sum_loss


batch_size = 32768
height = 128
width = 64


def get_inputs():
    x = torch.randn(batch_size, height, width)
    y = torch.randn(batch_size, height, width)
    return [x, y]


def get_init_inputs():
    return []