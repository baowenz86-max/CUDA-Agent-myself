import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, channel, num_cats):
        super().__init__()
        self.channel = channel
        self.num_cats = num_cats

    def forward(self, x):
        inf_mask = torch.isinf(x)
        x = torch.cat([x, inf_mask], dim=self.channel)
        x = torch.cat([x, torch.zeros_like(x)], dim=self.channel)
        return x


batch_size = 16
height = 512
width = 32
concat_dim = 1  # Separated from input channel size to avoid dimension conflict
num_cats = 3
input_channel = 128  # Increased input channel size to boost computation


def get_inputs():
    return [torch.randn(batch_size, height, width, input_channel)]


def get_init_inputs():
    return [concat_dim, num_cats]