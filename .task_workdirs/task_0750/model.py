import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.softmax = F.softmax
        self.cross_entropy = F.cross_entropy

    def forward(self, x, target):
        x = self.softmax(x, dim=1)
        x = self.cross_entropy(x, target)
        return x


batch_size = 512
height, width = 128, 128  # Increased spatial dimensions to increase computation
num_classes = 10


def get_inputs():
    x = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(num_classes, (batch_size, height, width))
    return [x, target]


def get_init_inputs():
    return [num_classes]