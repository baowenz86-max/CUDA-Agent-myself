import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_channels, num_bins, group_num, eps, mean_shape, var_shape):
        super().__init__()
        self.group_norm = F.group_norm
        self.batch_norm = nn.BatchNorm2d(num_channels, eps=eps)
        self.bins = nn.Parameter(torch.histc(torch.randn(1, num_channels, 32, 32), bins=num_bins, min=0, max=1))
        self.group_num = group_num
        self.mean_shape = mean_shape
        self.var_shape = var_shape

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.group_norm(x, self.group_num)
        x = torch.histc(x, bins=self.bins.shape[0], min=0, max=1)
        return x


batch_size = 512
num_channels = 3
num_bins = 10
group_num = 3
eps = 1e-05
mean_shape = (num_channels,)
var_shape = mean_shape


def get_inputs():
    x = torch.randn(batch_size, num_channels, 64, 64)
    return [x]


def get_init_inputs():
    return [num_channels, num_bins, group_num, eps, mean_shape, var_shape]