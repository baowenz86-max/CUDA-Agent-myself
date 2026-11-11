import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features, eps, momentum, affine, track_running_stats, margin, p, sample_size):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(
            num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
        )
        self.margin = margin
        self.p = p

    def forward(self, x, input_anchor, input_positive, input_negative):
        weight = self.batchnorm.weight
        bias = self.batchnorm.bias
        running_mean = self.batchnorm.running_mean
        running_var = self.batchnorm.running_var
        eps = self.batchnorm.eps

        x = cuda_extension.batch_norm_eval_2d(x, weight, bias, running_mean, running_var, eps)
        loss = cuda_extension.triplet_margin_loss_sum(input_anchor, input_positive, input_negative, self.margin, self.p)
        return x, loss

    def debug_forward(self, *inputs):
        x, input_anchor, input_positive, input_negative = inputs
        weight = self.batchnorm.weight
        bias = self.batchnorm.bias
        running_mean = self.batchnorm.running_mean
        running_var = self.batchnorm.running_var
        eps = self.batchnorm.eps

        x = cuda_extension.batch_norm_eval_2d(x, weight, bias, running_mean, running_var, eps)
        yield 'step_001', x

        loss = cuda_extension.triplet_margin_loss_sum(input_anchor, input_positive, input_negative, self.margin, self.p)
        yield 'step_002', loss

        result = (x, loss)
        yield 'step_003', result
