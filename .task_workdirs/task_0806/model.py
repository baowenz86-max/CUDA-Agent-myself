import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_list):
        x = torch.stack(x_list)
        x = F.softmax(x, dim=0)
        return x


batch_size = 512
features = 3
height, width = 128, 128  # Increased from 32x32 to 128x128


def get_inputs():
    x_list = [torch.randn(batch_size, features, height, width) for _ in range(4)]
    return [x_list]


def get_init_inputs():
    return []