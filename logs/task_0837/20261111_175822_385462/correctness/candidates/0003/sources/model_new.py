import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value):
        super().__init__()
        self.adaptive_logsoftmax = nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value)

    def forward(self, x, target):
        head_weight = self.adaptive_logsoftmax.head.weight
        tail_0_weight = self.adaptive_logsoftmax.tail[0][0].weight
        tail_1_weight = self.adaptive_logsoftmax.tail[0][1].weight

        head_log_probs = cuda_extension.adaptive_logsoftmax_head_forward(x, head_weight)
        tail_log_probs = cuda_extension.adaptive_logsoftmax_tail_forward(x, tail_0_weight, tail_1_weight)
        log_probs = cuda_extension.adaptive_logsoftmax_combine(head_log_probs, tail_log_probs)
        return log_probs

    def debug_forward(self, *inputs):
        x, target = inputs
        head_weight = self.adaptive_logsoftmax.head.weight
        tail_0_weight = self.adaptive_logsoftmax.tail[0][0].weight
        tail_1_weight = self.adaptive_logsoftmax.tail[0][1].weight

        head_log_probs = cuda_extension.adaptive_logsoftmax_head_forward(x, head_weight)
        yield ('step_001', head_log_probs)

        tail_log_probs = cuda_extension.adaptive_logsoftmax_tail_forward(x, tail_0_weight, tail_1_weight)
        log_probs = cuda_extension.adaptive_logsoftmax_combine(head_log_probs, tail_log_probs)
        yield ('step_002', log_probs)
        yield ('step_003', log_probs)
