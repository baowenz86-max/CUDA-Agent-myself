import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value, smoothing, input_size):
        super().__init__()
        # 保存参数所需的模块，与实际 AdaptiveLogSoftmaxWithLoss 结构一致
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(
            in_features, n_classes, cutoffs, div_value, smoothing
        )
        self.input_size = input_size

    def forward(self, x, target):
        # Step 1: one_hot + float
        x_onehot = cuda_extension.one_hot_float(x, self.input_size)
        # Step 2: adaptive log softmax log_prob
        log_probs = cuda_extension.adaptive_log_softmax_log_prob(
            x_onehot,
            self.adaptive_log_softmax.head.weight,
            self.adaptive_log_softmax.head.bias,
            self.adaptive_log_softmax.tail[0][0].weight,
            self.adaptive_log_softmax.tail[0][1].weight,
            self.adaptive_log_softmax.tail[1][0].weight,
            self.adaptive_log_softmax.tail[1][1].weight,
            self.adaptive_log_softmax.cutoffs
        )
        return log_probs, target

    def debug_forward(self, *inputs):
        x, target = inputs
        # checkpoint step_001
        x_onehot = cuda_extension.one_hot_float(x, self.input_size)
        yield 'step_001', x_onehot

        # checkpoint step_002
        log_probs = cuda_extension.adaptive_log_softmax_log_prob(
            x_onehot,
            self.adaptive_log_softmax.head.weight,
            self.adaptive_log_softmax.head.bias,
            self.adaptive_log_softmax.tail[0][0].weight,
            self.adaptive_log_softmax.tail[0][1].weight,
            self.adaptive_log_softmax.tail[1][0].weight,
            self.adaptive_log_softmax.tail[1][1].weight,
            self.adaptive_log_softmax.cutoffs
        )
        yield 'step_002', log_probs

        # checkpoint step_003: final return
        yield 'step_003', (log_probs, target)
