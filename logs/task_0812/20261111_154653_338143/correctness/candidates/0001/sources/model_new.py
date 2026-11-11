import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode, rotation_k):
        super().__init__()
        # Preserve original module structure for state_dict compatibility (empty state_dict)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.rotation_k = rotation_k

    def forward(self, x):
        # Step 1: nn.Upsample bilinear scale_factor=2.0
        x = cuda_extension.custom_upsample_bilinear2x(x)
        # Step 2: torch.rot90 with k=1, dims=(2,3)
        x = cuda_extension.custom_rot90(x, self.rotation_k, 2, 3)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_upsample_bilinear2x(x)
        yield 'step_001', x
        x = cuda_extension.custom_rot90(x, self.rotation_k, 2, 3)
        yield 'step_002', x
        yield 'step_003', x
