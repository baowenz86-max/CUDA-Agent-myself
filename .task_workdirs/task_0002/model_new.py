import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, num_groups, num_features):
        super().__init__()
        self.num_groups = num_groups
        self.num_features = num_features

    def forward(self, x):
        x = cuda_extension.ceil_forward(x)
        x = cuda_extension.transpose_1_2_forward(x)
        x = cuda_extension.group_norm_forward(x, self.num_groups)
        return x
