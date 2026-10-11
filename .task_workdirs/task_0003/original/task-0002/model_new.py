import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, in_channels, num_groups, num_features):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features

    def forward(self, x):
        return cuda_extension.ceil_transpose_group_norm(x, self.num_groups)
