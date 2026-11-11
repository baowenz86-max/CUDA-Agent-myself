import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value):
        super().__init__()
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value)
        self.in_features = in_features

    def forward(self, x, target):
        log_probs = self.adaptive_log_softmax.log_prob(x)
        softmax_output = torch.softmax(log_probs, dim=-1)
        cloned_output = torch.clone(softmax_output)
        return cloned_output, log_probs, target


in_features = 768
n_classes = 10000
cutoffs = [5000]
div_value = 4.0
batch_size = 2048  # Increased from 128 to increase runtime


def get_inputs():
    x = torch.randn(batch_size, in_features)
    target = torch.randint(0, n_classes, (batch_size,))
    return [x, target]


def get_init_inputs():
    return [in_features, n_classes, cutoffs, div_value]