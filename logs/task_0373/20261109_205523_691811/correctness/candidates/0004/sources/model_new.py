import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        x_max = cuda_extension.custom_max_dim1(x)
        x_stacked = cuda_extension.custom_stack_dim1(x)
        return x_max, x_stacked
