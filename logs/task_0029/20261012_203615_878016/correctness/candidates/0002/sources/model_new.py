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
        # step_001: torch.detach(x) -> tensor (identity; detach is a no-op for correctness)
        x_detached = x
        yield "step_001", x_detached

        # step_003: final result [torch.max(xi, dim=1) for xi in torch.unbind(x, dim=-1)]
        max_list = cuda_extension.custom_unbind_max(x)
        yield "step_003", max_list

        # step_004: return x
        yield "step_004", max_list
        return
