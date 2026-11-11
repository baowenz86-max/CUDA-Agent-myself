import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_channels, num_classes, loss_reduction="mean"):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.loss_reduction = loss_reduction

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        loss = F.smooth_l1_loss(x, torch.zeros_like(x), reduction=self.loss_reduction)
        return loss


def get_init_inputs():
    return [3, 2, "mean"]


batch_size = 512
channels = 10  # Increased from 1 to increase tensor size
height = 128
width = 128


def get_inputs():
    return [torch.randn(batch_size, channels, height, width), torch.randn(batch_size, channels, height, width)]