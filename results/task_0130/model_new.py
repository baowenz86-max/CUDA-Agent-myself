import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, anchor, positive, negative):
        anchor_mean = cuda_extension.adaptive_avg_pool2d_squeeze(anchor)
        positive_mean = cuda_extension.adaptive_avg_pool2d_squeeze(positive)
        negative_mean = cuda_extension.adaptive_avg_pool2d_squeeze(negative)
        loss = cuda_extension.triplet_margin_loss(anchor_mean, positive_mean, negative_mean)
        return loss

    def debug_forward(self, anchor, positive, negative):
        anchor_mean = cuda_extension.adaptive_avg_pool2d_squeeze(anchor)
        yield 'step_001', anchor_mean
        positive_mean = cuda_extension.adaptive_avg_pool2d_squeeze(positive)
        yield 'step_002', positive_mean
        negative_mean = cuda_extension.adaptive_avg_pool2d_squeeze(negative)
        yield 'step_003', negative_mean
        loss = cuda_extension.triplet_margin_loss(anchor_mean, positive_mean, negative_mean)
        yield 'step_004', loss
        yield 'step_005', loss
