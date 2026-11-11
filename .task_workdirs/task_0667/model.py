import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_shape1, input_shape2, target_shape):
        super().__init__()
        self.input_shape1 = input_shape1
        self.input_shape2 = input_shape2
        self.target_shape = target_shape

    def forward(self, x1, x2, target):
        x = torch.cat((x1, x2), dim=1)
        loss = F.mse_loss(x, target)
        return loss


batch_size = 512
channels1 = 16
channels2 = 16
height, width = 128, 128  # Increased from 32x32 to 128x128
target_channels = channels1 + channels2


def get_inputs():
    x1 = torch.randn(batch_size, channels1, height, width)
    x2 = torch.randn(batch_size, channels2, height, width)
    target = torch.randn(batch_size, target_channels, height, width)
    return [x1, x2, target]


def get_init_inputs():
    input_shape1 = (batch_size, channels1, height, width)
    input_shape2 = (batch_size, channels2, height, width)
    target_shape = (batch_size, target_channels, height, width)
    return [input_shape1, input_shape2, target_shape]