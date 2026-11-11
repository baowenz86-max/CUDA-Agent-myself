import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features, n_classes, cutoffs, div_value):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = list(cutoffs)
        self.div_value = div_value
        self.head_size = self.cutoffs[0] + n_classes - self.cutoffs[-1]

        # Create parameters matching reference state_dict.
        # All Linear layers have bias=False.
        self.adaptive_log_softmax = nn.Module()
        self.adaptive_log_softmax.head = nn.Module()
        self.adaptive_log_softmax.head.weight = nn.Parameter(
            torch.empty(self.head_size, in_features)
        )
        self.adaptive_log_softmax.tail = nn.ModuleList()
        for i, cutoff in enumerate(self.cutoffs):
            if i == 0:
                hidden_size = in_features // 4
            else:
                hidden_size = in_features // 16
            tail_seq = nn.Sequential()
            tail_seq.add_module('0', nn.Linear(in_features, hidden_size, bias=False))
            tail_seq.add_module('1', nn.Linear(hidden_size, self.cutoffs[1] - self.cutoffs[0], bias=False))
            self.adaptive_log_softmax.tail.append(tail_seq)

        # Move parameters to match reference state_dict keys.
        self._init_parameters()

    def _init_parameters(self):
        # Initialize weights using the same scheme as PyTorch's AdaptiveLogSoftmaxWithLoss.
        # Head weight: normal distribution with std 0.1
        nn.init.normal_(self.adaptive_log_softmax.head.weight, 0, 0.1)
        for tail in self.adaptive_log_softmax.tail:
            nn.init.normal_(tail[0].weight, 0, 0.1)
            nn.init.normal_(tail[1].weight, 0, 0.1)

    def forward(self, x, target):
        # Ensure x and target are on the same device as parameters and are contiguous.
        x = x.cuda() if not x.is_cuda else x
        target = target.cuda() if not target.is_cuda else target

        # Convert parameters to contiguous tensors on the fly.
        head_weight = self.adaptive_log_softmax.head.weight.detach().contiguous()
        tail0_l1_weight = self.adaptive_log_softmax.tail[0][0].weight.detach().contiguous()
        tail0_l2_weight = self.adaptive_log_softmax.tail[0][1].weight.detach().contiguous()
        tail1_l1_weight = self.adaptive_log_softmax.tail[1][0].weight.detach().contiguous()
        tail1_l2_weight = self.adaptive_log_softmax.tail[1][1].weight.detach().contiguous()

        # Call the custom fused operator that mirrors AdaptiveLogSoftmaxWithLoss.__call__.
        # It returns (log_prob_output, asm_output) where asm_output contains
        # (per_sample_loss, mean_loss).
        log_prob_output, asm_output = cuda_extension.custom_adaptive_log_softmax_with_loss(
            x, target,
            head_weight,
            tail0_l1_weight,
            tail0_l2_weight,
            tail1_l1_weight,
            tail1_l2_weight,
            self.cutoffs[0],
            self.cutoffs[1],
            self.n_classes,
            self.div_value,
        )

        # Compute abs(log_prob_output) using custom op.
        out = cuda_extension.custom_abs(log_prob_output)

        return out, asm_output

    def debug_forward(self, *inputs):
        # Must follow same logic as forward but yield checkpoint values.
        x, target = inputs
        # Move to cuda
        x = x.cuda() if not x.is_cuda else x
        target = target.cuda() if not target.is_cuda else target

        head_weight = self.adaptive_log_softmax.head.weight.detach().contiguous()
        tail0_l1_weight = self.adaptive_log_softmax.tail[0][0].weight.detach().contiguous()
        tail0_l2_weight = self.adaptive_log_softmax.tail[0][1].weight.detach().contiguous()
        tail1_l1_weight = self.adaptive_log_softmax.tail[1][0].weight.detach().contiguous()
        tail1_l2_weight = self.adaptive_log_softmax.tail[1][1].weight.detach().contiguous()

        # Step 001: log_probs
        log_probs, _ = cuda_extension.custom_adaptive_log_softmax_with_loss(
            x, target,
            head_weight,
            tail0_l1_weight,
            tail0_l2_weight,
            tail1_l1_weight,
            tail1_l2_weight,
            self.cutoffs[0],
            self.cutoffs[1],
            self.n_classes,
            self.div_value,
        )
        yield 'step_001', log_probs

        # Step 002: out = abs(log_probs)
        out = cuda_extension.custom_abs(log_probs)
        yield 'step_002', out

        # Step 003: final return
        _, asm_output = cuda_extension.custom_adaptive_log_softmax_with_loss(
            x, target,
            head_weight,
            tail0_l1_weight,
            tail0_l2_weight,
            tail1_l1_weight,
            tail1_l2_weight,
            self.cutoffs[0],
            self.cutoffs[1],
            self.n_classes,
            self.div_value,
        )
        final_tuple = (out, asm_output)
        yield 'step_003', final_tuple
