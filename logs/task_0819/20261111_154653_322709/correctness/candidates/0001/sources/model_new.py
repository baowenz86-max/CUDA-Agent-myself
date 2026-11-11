import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value):
        super().__init__()
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(in_features, n_classes, cutoffs, div_value)

    def forward(self, x, target):
        log_probs = cuda_extension.adaptive_logsoftmax_logprob(
            x, self.adaptive_log_softmax.head.weight,
            self.adaptive_log_softmax.tail[0][0].weight,
            self.adaptive_log_softmax.tail[0][1].weight,
            self.adaptive_log_softmax.tail[1][0].weight,
            self.adaptive_log_softmax.tail[1][1].weight,
            self.adaptive_log_softmax.cutoffs,
            self.adaptive_log_softmax.div_value
        )
        out = cuda_extension.absolute(log_probs)
        loss_output = cuda_extension.adaptive_logsoftmax_with_loss(
            self.adaptive_log_softmax.head.weight,
            x, target,
            self.adaptive_log_softmax.tail[0][0].weight,
            self.adaptive_log_softmax.tail[0][1].weight,
            self.adaptive_log_softmax.tail[1][0].weight,
            self.adaptive_log_softmax.tail[1][1].weight,
            self.adaptive_log_softmax.cutoffs,
            self.adaptive_log_softmax.div_value
        )
        return out, loss_output

    def debug_forward(self, x, target):
        log_probs = cuda_extension.adaptive_logsoftmax_logprob(
            x, self.adaptive_log_softmax.head.weight,
            self.adaptive_log_softmax.tail[0][0].weight,
            self.adaptive_log_softmax.tail[0][1].weight,
            self.adaptive_log_softmax.tail[1][0].weight,
            self.adaptive_log_softmax.tail[1][1].weight,
            self.adaptive_log_softmax.cutoffs,
            self.adaptive_log_softmax.div_value
        )
        yield "step_001", log_probs
        out = cuda_extension.absolute(log_probs)
        yield "step_002", out
        loss_output = cuda_extension.adaptive_logsoftmax_with_loss(
            self.adaptive_log_softmax.head.weight,
            x, target,
            self.adaptive_log_softmax.tail[0][0].weight,
            self.adaptive_log_softmax.tail[0][1].weight,
            self.adaptive_log_softmax.tail[1][0].weight,
            self.adaptive_log_softmax.tail[1][1].weight,
            self.adaptive_log_softmax.cutoffs,
            self.adaptive_log_softmax.div_value
        )
        yield "step_003", (out, loss_output)
