import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(in_features, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.leaky_relu_forward(x, negative_slope=0.01)
        x = cuda_extension.instance_norm_forward(
            x,
            self.instance_norm.weight,
            self.instance_norm.bias,
            self.instance_norm.eps,
        )
        return x

    def debug_forward(self, x: torch.Tensor):
        y = cuda_extension.leaky_relu_forward(x, negative_slope=0.01)
        yield 'step_001', y
        z = cuda_extension.instance_norm_forward(
            y,
            self.instance_norm.weight,
            self.instance_norm.bias,
            self.instance_norm.eps,
        )
        yield 'step_002', z
        yield 'step_003', z
