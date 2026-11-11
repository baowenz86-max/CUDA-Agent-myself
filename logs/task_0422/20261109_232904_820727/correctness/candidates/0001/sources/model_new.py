import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        max_val, _ = cuda_extension.custom_max_dim(x, 1, True)
        loss = cuda_extension.custom_mse_loss(max_val, y)
        return loss

    def debug_forward(self, x: torch.Tensor, y: torch.Tensor):
        max_val, _ = cuda_extension.custom_max_dim(x, 1, True)
        yield 'step_001', (max_val, _)
        loss = cuda_extension.custom_mse_loss(max_val, y)
        yield 'step_002', loss
        yield 'step_003', loss
