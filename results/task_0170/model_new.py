import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, pool_size: tuple, min_val: float, max_val: float):
        super().__init__()
        # Store configuration for custom ops; no learnable parameters to keep state_dict empty
        self.pool_size = pool_size
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: AvgPool2d with kernel=pool_size, stride=pool_size, padding=0
        x = cuda_extension.avgpool2d_forward(
            x,
            self.pool_size[0],
            self.pool_size[1],
            self.pool_size[0],
            self.pool_size[1],
            0,
            0
        )
        # Step 2: Hardtanh with min_val and max_val
        x = cuda_extension.hardtanh_forward(x, self.min_val, self.max_val)
        return x

    def debug_forward(self, x: torch.Tensor):
        # Yield each intermediate checkpoint just like reference trace
        x = cuda_extension.avgpool2d_forward(
            x,
            self.pool_size[0],
            self.pool_size[1],
            self.pool_size[0],
            self.pool_size[1],
            0,
            0
        )
        yield "step_001", x
        x = cuda_extension.hardtanh_forward(x, self.min_val, self.max_val)
        yield "step_002", x
        yield "step_003", x
