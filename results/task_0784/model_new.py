import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=3)

    def forward(self, x):
        return cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            3,
            1e-05
        )

    def debug_forward(self, *inputs):
        x = inputs[0]
        output = cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            3,
            1e-05
        )
        yield ('step_001', output)
