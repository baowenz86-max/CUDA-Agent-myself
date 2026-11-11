import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        result = torch.equal(x, y)
        result = F.l1_loss(x, y)
        return result


batch_size = 2048  # Increased batch size to increase computation
height, width = 512, 128


def get_inputs():
    x = torch.randn(batch_size, height, width)
    y = torch.randn(batch_size, height, width)
    return [x, y]


def get_init_inputs():
    return []