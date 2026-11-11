import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, cat_dim):
        super(ModelNew, self).__init__()
        self.cat_dim = cat_dim

    def forward(self, x1, x2):
        x1 = cuda_extension.custom_cat(x1, x2, self.cat_dim)
        x1 = cuda_extension.custom_rot90(x1, 1, 2, 3)
        return x1

    def debug_forward(self, *inputs):
        x1, x2 = inputs
        x1 = cuda_extension.custom_cat(x1, x2, self.cat_dim)
        yield "step_001", x1
        x1 = cuda_extension.custom_rot90(x1, 1, 2, 3)
        yield "step_002", x1
        yield "step_003", x1
