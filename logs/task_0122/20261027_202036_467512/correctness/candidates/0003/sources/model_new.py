import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, n_classes, cutoffs, div_value):
        super().__init__()
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(input_size, n_classes, cutoffs, div_value)

    def forward(self, x, targets, k):
        if not x.is_cuda:
            x = x.cuda()
        log_probs = cuda_extension.adaptive_log_softmax_log_prob(
            x.contiguous(),
            self.adaptive_log_softmax.head.weight.contiguous(),
            self.adaptive_log_softmax.tail[0][0].weight.contiguous(),
            self.adaptive_log_softmax.tail[0][1].weight.contiguous(),
            self.adaptive_log_softmax.tail[1][0].weight.contiguous(),
            self.adaptive_log_softmax.tail[1][1].weight.contiguous(),
            self.adaptive_log_softmax.cutoffs,
            self.adaptive_log_softmax.div_value
        )
        values, indices = cuda_extension.topk(log_probs, k)
        topk_result = torch.return_types.topk((values, indices))
        return topk_result.values, topk_result.indices

    def debug_forward(self, x, targets, k):
        if not x.is_cuda:
            x = x.cuda()
        log_probs = cuda_extension.adaptive_log_softmax_log_prob(
            x.contiguous(),
            self.adaptive_log_softmax.head.weight.contiguous(),
            self.adaptive_log_softmax.tail[0][0].weight.contiguous(),
            self.adaptive_log_softmax.tail[0][1].weight.contiguous(),
            self.adaptive_log_softmax.tail[1][0].weight.contiguous(),
            self.adaptive_log_softmax.tail[1][1].weight.contiguous(),
            self.adaptive_log_softmax.cutoffs,
            self.adaptive_log_softmax.div_value
        )
        yield 'step_001', log_probs

        values, indices = cuda_extension.topk(log_probs, k)
        topk_result = torch.return_types.topk((values, indices))
        yield 'step_002', (topk_result.values, topk_result.indices)

        yield 'step_003', (topk_result.values, topk_result.indices)
