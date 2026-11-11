import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps, momentum, affine, track_running_stats, margin, p, sample_size):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=p, reduction="sum")

    def forward(self, x, input_anchor, input_positive, input_negative):
        x = cuda_extension.batchnorm_inference(
            x,
            self.batchnorm.weight,
            self.batchnorm.bias,
            self.batchnorm.running_mean,
            self.batchnorm.running_var,
            self.batchnorm.eps
        )
        loss = cuda_extension.triplet_margin_loss_sum(
            input_anchor,
            input_positive,
            input_negative,
            self.triplet_loss.margin,
            self.triplet_loss.p
        )
        return x, loss

    def debug_forward(self, x, input_anchor, input_positive, input_negative):
        x = cuda_extension.batchnorm_inference(
            x,
            self.batchnorm.weight,
            self.batchnorm.bias,
            self.batchnorm.running_mean,
            self.batchnorm.running_var,
            self.batchnorm.eps
        )
        yield "step_001", x
        loss = cuda_extension.triplet_margin_loss_sum(
            input_anchor,
            input_positive,
            input_negative,
            self.triplet_loss.margin,
            self.triplet_loss.p
        )
        yield "step_002", loss
        yield "step_003", (x, loss)
