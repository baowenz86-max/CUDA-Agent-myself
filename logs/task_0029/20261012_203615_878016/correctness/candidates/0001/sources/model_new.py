import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        return cuda_extension.custom_unbind_max(x)

    def debug_forward(self, x):
        # step_001: torch.detach(x) -> tensor
        x_detached = x  # detach is not necessary for correctness, but keep checkpoint
        yield "step_001", x_detached

        # step_002: torch.unbind(x, dim=-1) -> tuple of tensors
        # We must reproduce tuple of tensors; custom_op could provide it, but we need exact structure.
        # We temporarily use torch.unbind only for debug checkpoint generation, but actual computation uses custom op.
        # Since debug forward must not use torch ops for computation, but this is just for checkpoint structure,
        # we can create placeholders? However checkpoint requires actual values.
        # To comply with "debug_forward cannot change calculation formula", we compute using custom op for final result,
        # and for step_002 we need the unbind tuple. We can call a custom op to perform unbind only (maybe same op with flag).
        # We define custom_unbind_max to also return intermediate unbind result? But simpler: custom_unbind_max returns final list,
        # and we can derive unbind tuple by using custom op custom_unbind (if available) or by calling custom_unbind_max and then reconstructing?
        # To avoid extra ops, we can call cuda_extension.custom_unbind(x) which we assume will be implemented later.
        # Since we are only allowed to call custom ops that will be implemented, we define two ops: custom_unbind and custom_unbind_max.
        # For now, we assume they will exist. We'll use them.
        unbind_tuple = cuda_extension.custom_unbind(x)
        yield "step_002", unbind_tuple

        # step_003: list comprehension [torch.max(xi, dim=1) for xi in x]
        max_list = cuda_extension.custom_unbind_max(x)
        yield "step_003", max_list

        # step_004: return x
        yield "step_004", max_list
        return
