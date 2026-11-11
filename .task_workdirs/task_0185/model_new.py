import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Step 1: mean over channel dimension keeping the dimension
        x = cuda_extension.mean_dim1_keepdim(x)
        # Step 2: tile along channel dimension by factor 8
        x = cuda_extension.tile_dim1(x, 8)
        return x

    def debug_forward(self, x):
        # Recompute steps identically to forward for checkpoint tracking.
        x = cuda_extension.mean_dim1_keepdim(x)
        yield ('step_001', x)
        x = cuda_extension.tile_dim1(x, 8)
        yield ('step_002', x)
        yield ('step_003', x)
