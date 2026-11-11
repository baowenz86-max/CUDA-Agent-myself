import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, norm_type, num_channels):
        super().__init__()
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(num_channels)
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm2d(num_channels)
        elif norm_type == "layer":
            self.norm = nn.LayerNorm(num_channels)
        else:
            raise ValueError("Invalid normalization type. Choose 'batch', 'instance', or 'layer'.")

    def forward(self, x):
        return cuda_extension.norm_forward(
            x,
            self.norm.weight,
            self.norm.bias,
            self.norm.running_mean,
            self.norm.running_var,
            self.norm.eps,
            self.norm.momentum,
            False if self.training else True,
        )

    def debug_forward(self, x):
        y = self.forward(x)
        yield "step_001", y
