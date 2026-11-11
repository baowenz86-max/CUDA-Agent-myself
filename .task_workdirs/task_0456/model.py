import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features, eps, momentum, affine, track_running_stats, margin, p, sample_size):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p, reduction="sum")

    def forward(self, x, input_anchor, input_positive, input_negative):
        x = self.batchnorm(x)
        loss = self.triplet_loss(input_anchor, input_positive, input_negative)
        return x, loss


num_features = 768
eps = 1e-05
momentum = 0.1
affine = True
track_running_stats = True
margin = 1.0
p = 2
sample_size = 16


def get_inputs():
    x = torch.randn(16, num_features, 64, 64)
    input_anchor = torch.randn(16, num_features, 64, 64)
    input_positive = torch.randn(16, num_features, 64, 64)
    input_negative = torch.randn(16, num_features, 64, 64)
    return [x, input_anchor, input_positive, input_negative]


def get_init_inputs():
    return [num_features, eps, momentum, affine, track_running_stats, margin, p, sample_size]