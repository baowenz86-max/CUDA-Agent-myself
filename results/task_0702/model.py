import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        x = torch.sum(x, dim=1)  # sum across channels
        x = F.log_softmax(x, dim=0)  # apply log_softmax across the correct dimension
        return x


batch_size = 640  # Increased batch size to increase computation
height, width = 128, 1024
num_classes = 10  # example number of classes


def get_inputs():
    return [torch.randn(batch_size, 1, height, width)]


def get_init_inputs():
    return [num_classes]