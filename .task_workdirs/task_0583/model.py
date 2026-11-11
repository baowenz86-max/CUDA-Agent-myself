import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Check if any elements in x are inf
        inf_mask = torch.isinf(x)
        # If there are any inf elements, replace them with a large negative number
        x = torch.where(inf_mask, -torch.inf, x)
        x = self.log_softmax(x)
        return x


batch_size = 512
height, width = 128, 128  # Increased from 32x32 to 128x128 to increase computation
input_channels = 16
num_classes = 10


def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]


def get_init_inputs():
    return [input_channels, num_classes]