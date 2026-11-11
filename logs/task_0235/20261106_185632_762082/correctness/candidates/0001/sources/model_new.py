import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, num_features, eps, affine, num_groups):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.num_groups = num_groups
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        import cuda_extension
        gn_out = cuda_extension.custom_group_norm(
            x, self.weight, self.bias, self.num_groups, self.num_features, self.eps
        )
        return cuda_extension.custom_permute_sum(gn_out)

    def debug_forward(self, x):
        import cuda_extension
        gn_out = cuda_extension.custom_group_norm(
            x, self.weight, self.bias, self.num_groups, self.num_features, self.eps
        )
        out = cuda_extension.custom_permute_sum(gn_out)
        yield 'step_001', out
