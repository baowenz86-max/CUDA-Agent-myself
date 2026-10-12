import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        # Obtain the torch.max return type once; it is stable across calls.
        dummy = torch.zeros(1, device='cuda')
        self._max_return_type = type(torch.max(dummy, 0))

    def forward(self, x):
        # x shape: (B, C, H, W) with C=3, W=512
        results = cuda_extension.custom_unbind_max(x)
        # results is a list of W pairs (values, indices), each shape (B,H)
        # Convert each pair to torch.return_types.max to match reference type.
        typed_results = [
            self._max_return_type((values, indices))
            for values, indices in results
        ]
        return typed_results

    def debug_forward(self, x):
        # Checkpoint step_001: x = torch.detach(x)
        x_detached = cuda_extension.detach_forward(x)
        yield 'step_001', x_detached

        # Checkpoint step_002: x = torch.unbind(x, dim=-1)
        # Use custom unbind op returning a tuple of W tensors, each shape (B,C,H)
        unbind_tuple = cuda_extension.custom_unbind(x_detached)
        yield 'step_002', unbind_tuple

        # Checkpoint step_003: x = [torch.max(xi, dim=1) for xi in x]
        # Use custom op that computes max for each slice; returns list of W pairs
        max_pairs = cuda_extension.custom_unbind_max_weighted(x_detached)
        typed_pairs = [
            self._max_return_type((values, indices))
            for values, indices in max_pairs
        ]
        yield 'step_003', typed_pairs

        # Final return
        yield 'step_004', typed_pairs
