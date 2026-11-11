import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        x = torch.abs(x)
        x = F.one_hot(torch.argmax(x, dim=1), num_classes=self.num_classes)
        return x


batch_size = 512
height, width = 128, 128  # Increased from 32x32 to 128x128
num_classes = 10


def get_inputs():
    return [torch.randn(batch_size, num_classes, height, width)]


def get_init_inputs():
    return [num_classes]