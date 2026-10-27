import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, n_classes, cutoffs, div_value):
        super().__init__()
        # Store parameters as in nn.Module, preserving state_dict keys
        self.adaptive_log_softmax = nn.AdaptiveLogSoftmaxWithLoss(input_size, n_classes, cutoffs, div_value)

    def forward(self, x, targets, k):
        # Move input to CUDA if needed (harness may provide CPU tensors)
        if not x.is_cuda:
            x = x.cuda()
        # Step 1: log_probs
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
        # Step 2: topk
        topk_result = cuda_extension.topk(log_probs, k)
        # topk_result is a torch.return_types.topk object containing values and indices
        # We return a plain tuple as in reference
        return topk_result.values, topk_result.indices

    def debug_forward(self, x, targets, k):
        # Same computation but yields intermediate checkpoints
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

        topk_result = cuda_extension.topk(log_probs, k)
        yield 'step_002', (topk_result.values, topk_result.indices)

        # Final return
        yield 'step_003', (topk_result.values, topk_result.indices)
