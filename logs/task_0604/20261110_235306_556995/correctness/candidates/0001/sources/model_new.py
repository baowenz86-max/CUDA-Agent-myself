import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return cuda_extension.abs_relu6_forward(x)

    def debug_forward(self, x):
        # Step 1: torch.abs(x)
        x_abs = cuda_extension.abs_relu6_forward(x)
        yield ('step_001', x_abs)
        # Step 2: nn.ReLU6()(x)
        x_relu6 = cuda_extension.abs_relu6_forward(x_abs)
        yield ('step_002', x_relu6)
        # Step 3: return x
        yield ('step_003', x_relu6)
