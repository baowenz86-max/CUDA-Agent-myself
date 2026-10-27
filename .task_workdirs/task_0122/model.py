import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, n_classes, cutoffs, div_value):
        super().__init__()
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(input_size, n_classes, cutoffs, div_value)

    def forward(self, x, targets, k):
        log_probs = self.adaptive_log_softmax.log_prob(x)
        topk_values, topk_indices = torch.topk(log_probs, k)
        return topk_values, topk_indices


input_size = 768
n_classes = 10000
cutoffs = [5000, 8000]
div_value = 4.0
batch_size = 2048  # Increased from 128 to increase runtime
k = 5


def get_inputs():
    x = torch.randn(batch_size, input_size)
    targets = torch.randint(0, n_classes, (batch_size,))
    return [x, targets, k]


def get_init_inputs():
    return [input_size, n_classes, cutoffs, div_value]