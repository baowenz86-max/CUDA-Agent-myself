import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape, apply_sigmoid):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid

    def forward(self, x):
        x = torch.sum(x, dim=1, keepdim=True)  # Assuming input has dimension [batch_size, channels, height, width]
        if self.apply_sigmoid:
            x = torch.sigmoid(x)
        return x


batch_size = 512
channels = 3
height, width = 512, 512  # Increased from 32x32 to 512x512
apply_sigmoid = True


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [None, apply_sigmoid]