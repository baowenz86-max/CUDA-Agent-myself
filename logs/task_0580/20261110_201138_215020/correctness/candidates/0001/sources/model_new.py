import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Step 1: max over dim=1 with keepdim=True, discarding indices
        x, _ = cuda_extension.max_dim(x, dim=1, keepdim=True)
        # Step 2: binary cross entropy with logits against random target of same shape
        target = torch.randn_like(x)
        loss = cuda_extension.binary_cross_entropy_with_logits(x, target)
        return loss

    def debug_forward(self, x):
        x, _ = cuda_extension.max_dim(x, dim=1, keepdim=True)
        yield 'step_001', x
        target = torch.randn_like(x)
        loss = cuda_extension.binary_cross_entropy_with_logits(x, target)
        yield 'step_002', loss
