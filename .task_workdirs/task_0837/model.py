import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value):
        super().__init__()
        self.adaptive_logsoftmax = nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value)

    def forward(self, x, target):
        log_probs = self.adaptive_logsoftmax.log_prob(x)
        log_probs = torch.unsqueeze(log_probs, 1)
        return log_probs


batch_size = 2048  # Increased from 128 to increase computation
in_features = 512
n_classes = 10000
cutoffs = [5000]
div_value = 2.0
seq_len = 1


def get_inputs():
    x = torch.randn(batch_size, in_features)
    target = torch.randint(0, n_classes, (batch_size,))
    return [x, target]


def get_init_inputs():
    return [in_features, n_classes, cutoffs, div_value]