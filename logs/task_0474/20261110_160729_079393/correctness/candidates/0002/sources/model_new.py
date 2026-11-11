import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value):
        super().__init__()
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value)
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs
        self.div_value = div_value

    def forward(self, x, target):
        # Step 1: compute log_probs using AdaptiveLogSoftmaxWithLoss.log_prob
        log_probs = cuda_extension.adaptive_log_softmax(
            x,
            self.adaptive_log_softmax.head.weight,
            self.adaptive_log_softmax.tail[0][0].weight,
            self.adaptive_log_softmax.tail[0][1].weight,
            self.cutoffs,
            self.div_value
        )
        # Step 2: softmax over last dim
        softmax_output = cuda_extension.custom_softmax(log_probs, dim=-1)
        # Step 3: clone softmax output
        cloned_output = cuda_extension.custom_clone(softmax_output)
        return cloned_output, log_probs, target

    def debug_forward(self, x, target):
        # replicate step-by-step for debugging
        log_probs = cuda_extension.adaptive_log_softmax(
            x,
            self.adaptive_log_softmax.head.weight,
            self.adaptive_log_softmax.tail[0][0].weight,
            self.adaptive_log_softmax.tail[0][1].weight,
            self.cutoffs,
            self.div_value
        )
        yield 'step_001', log_probs
        softmax_output = cuda_extension.custom_softmax(log_probs, dim=-1)
        yield 'step_002', softmax_output
        cloned_output = cuda_extension.custom_clone(softmax_output)
        yield 'step_003', cloned_output
        yield 'step_004', (cloned_output, log_probs, target)

    def _apply(self, fn):
        super()._apply(fn)
        self.adaptive_log_softmax._apply(fn)
        return self
