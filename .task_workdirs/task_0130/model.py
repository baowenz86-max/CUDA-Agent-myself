import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.triplet_loss = nn.TripletMarginLoss()

    def forward(self, anchor, positive, negative):
        anchor_mean = self.mean(anchor).squeeze()
        positive_mean = self.mean(positive).squeeze()
        negative_mean = self.mean(negative).squeeze()
        loss = self.triplet_loss(anchor_mean, positive_mean, negative_mean)
        return loss


batch_size = 512
channels = 128
height = 112
width = 112


def get_inputs():
    anchor = torch.randn(batch_size, channels, height, width)
    positive = torch.randn(batch_size, channels, height, width)
    negative = torch.randn(batch_size, channels, height, width)
    return [anchor, positive, negative]


def get_init_inputs():
    return []