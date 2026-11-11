import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, feature_size, num_groups, eps=1e-5, affine=True):
        super().__init__()
        self.group_norm = torch.nn.GroupNorm(num_groups=num_groups, num_channels=feature_size, eps=eps, affine=affine)

    def forward(self, x):
        x = cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        out = cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps
        )
        yield 'step_001', out
        yield 'step_002', out
