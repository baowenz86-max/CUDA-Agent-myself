import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels, p):
        super().__init__()
        self.p = p

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = F.normalize(x, p=self.p, dim=1)
        return x


batch_size = 512
in_channels_1, in_channels_2 = 3, 64
out_channels = 128
p = 2


def get_inputs():
    x1 = torch.randn(batch_size, in_channels_1, 128, 128)
    x2 = torch.randn(batch_size, in_channels_2, 128, 128)
    return [x1, x2]


def get_init_inputs():
    return [in_channels_1, in_channels_2, out_channels, p]