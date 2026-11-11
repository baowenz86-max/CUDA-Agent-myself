import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: max over channels, keepdim=True
        max_result = cuda_extension.custom_max_dim1(x)  # returns torch.return_types.max
        x = max_result.values
        # Step 2: ReLU
        x = cuda_extension.custom_relu(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        # Must yield (checkpoint_id, value) for each checkpoint in order
        # Step 001: torch.max returns tuple (values, indices)
        max_result = cuda_extension.custom_max_dim1(x)
        yield ('step_001', (max_result.values, max_result.indices))
        # Step 002: ReLU on values
        x = cuda_extension.custom_relu(max_result.values)
        yield ('step_002', x)
        # Step 003: final return
        yield ('step_003', x)
