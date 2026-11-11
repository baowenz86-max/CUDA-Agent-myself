import torch
import torch.nn as nn
import torch.nn.functional as F
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value, smoothing, input_size):
        super().__init__()
        self.input_size = input_size
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs
        self.div_value = div_value
        self.smoothing = smoothing

        # 建立与 nn.AdaptiveLogSoftmaxWithLoss 相同的参数结构，保证 state_dict 键一致
        self.adaptive_log_softmax = nn.Module()
        self.adaptive_log_softmax.head = nn.Linear(in_features, cutoffs[0] + 2, bias=True)
        self.adaptive_log_softmax.tail = nn.ModuleList()

        # 第一个 tail cluster：输入 in_features -> 256 -> 500
        first_tail = nn.Sequential(
            nn.Linear(in_features, 256, bias=False),
            nn.Linear(256, 500, bias=False)
        )
        self.adaptive_log_softmax.tail.append(first_tail)

        # 第二个 tail cluster：输入 in_features -> 64 -> 4200
        second_tail = nn.Sequential(
            nn.Linear(in_features, 64, bias=False),
            nn.Linear(64, 4200, bias=False)
        )
        self.adaptive_log_softmax.tail.append(second_tail)

        self.head_num_classes = cutoffs[0] + 2

    def forward(self, x, target):
        # 仅使用自定义 CUDA 扩展调用
        x = cuda_extension.one_hot_forward(x, self.input_size)
        log_probs = cuda_extension.adaptive_log_softmax_log_prob_forward(
            x,
            self.adaptive_log_softmax.head.weight,
            self.adaptive_log_softmax.head.bias,
            self.adaptive_log_softmax.tail[0][0].weight,
            self.adaptive_log_softmax.tail[0][1].weight,
            self.adaptive_log_softmax.tail[1][0].weight,
            self.adaptive_log_softmax.tail[1][1].weight,
            self.cutoffs,
            self.div_value,
            self.n_classes,
            self.head_num_classes
        )
        return log_probs, target

    def debug_forward(self, x, target):
        x_one_hot = cuda_extension.one_hot_forward(x, self.input_size)
        log_probs = cuda_extension.adaptive_log_softmax_log_prob_forward(
            x_one_hot,
            self.adaptive_log_softmax.head.weight,
            self.adaptive_log_softmax.head.bias,
            self.adaptive_log_softmax.tail[0][0].weight,
            self.adaptive_log_softmax.tail[0][1].weight,
            self.adaptive_log_softmax.tail[1][0].weight,
            self.adaptive_log_softmax.tail[1][1].weight,
            self.cutoffs,
            self.div_value,
            self.n_classes,
            self.head_num_classes
        )
        yield ('step_001', x_one_hot)
        yield ('step_002', log_probs)
        yield ('step_003', (log_probs, target))
