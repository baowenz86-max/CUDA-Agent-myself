import torch
import torch.nn as nn


class ModelNew(nn.Module):
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

    def debug_forward(self, x, input_anchor, input_positive, input_negative):
        x = self.batchnorm(x)
        yield "step_001", x
        loss = self.triplet_loss(input_anchor, input_positive, input_negative)
        yield "step_002", loss
        yield "step_003", (x, loss)