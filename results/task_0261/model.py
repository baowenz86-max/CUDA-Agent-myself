import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value, smoothing, input_size):
        super().__init__()
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value, smoothing)
        self.input_size = input_size

    def forward(self, x, target):
        x = F.one_hot(x, num_classes=self.input_size).float()
        log_probs = self.adaptive_log_softmax.log_prob(x)
        return log_probs, target


batch_size = 8192  # Increased to increase computation load
in_features = 1024  # Increased input feature size
n_classes = 5000  # Increased number of classes
cutoffs = [300, 800]
div_value = 4.0
smoothing = 0.1
input_size = 1024  # Matches in_features for valid input shape
target_size = batch_size


def get_inputs():
    x = torch.randint(0, input_size, (batch_size,))
    target = torch.randint(0, n_classes, (batch_size,))
    return [x, target]


def get_init_inputs():
    return [in_features, n_classes, cutoffs, div_value, smoothing, input_size]