import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, num_features, eps, affine, num_groups):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.num_groups = num_groups
        # Create a submodule to hold parameters with names matching reference GroupNorm
        self.group_norm = nn.Module()
        if affine:
            self.group_norm.weight = nn.Parameter(torch.ones(num_features))
            self.group_norm.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.group_norm.register_parameter('weight', None)
            self.group_norm.register_parameter('bias', None)

    def forward(self, x):
        import cuda_extension
        gn_out = cuda_extension.custom_group_norm(
            x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.num_features, self.eps
        )
        return cuda_extension.custom_permute_sum(gn_out)

    def debug_forward(self, x):
        import cuda_extension
        gn_out = cuda_extension.custom_group_norm(
            x, self.group_norm.weight, self.group_norm.bias, self.num_groups, self.num_features, self.eps
        )
        out = cuda_extension.custom_permute_sum(gn_out)
        yield 'step_001', out
