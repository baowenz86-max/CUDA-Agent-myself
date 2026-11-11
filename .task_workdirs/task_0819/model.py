import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value):
        super().__init__()
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value)

    def forward(self, x, target):
        log_probs = self.adaptive_log_softmax.log_prob(x)
        out = torch.abs(log_probs)
        return out, self.adaptive_log_softmax(x, target)


batch_size = 512  # Increased from 128 to increase computation load
in_features = 1024
n_classes = 20000
cutoffs = [10000, 15000]
div_value = 4.0


def get_inputs():
    x = torch.randn(batch_size, in_features)
    target = torch.randint(0, n_classes, (batch_size,))
    return [x, target]


def get_init_inputs():
    return [in_features, n_classes, cutoffs, div_value]