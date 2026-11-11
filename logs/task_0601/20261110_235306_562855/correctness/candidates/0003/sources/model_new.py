import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, cat_dim):
        super(ModelNew, self).__init__()
        self.cat_dim = cat_dim

    def forward(self, x1, x2):
        x1 = cuda_extension.concat_dim(x1, x2, self.cat_dim)
        x1 = cuda_extension.rot90_k1_dims23(x1)
        return x1

    def debug_forward(self, x1, x2):
        y = cuda_extension.concat_dim(x1, x2, self.cat_dim)
        yield "step_001", y
        z = cuda_extension.rot90_k1_dims23(y)
        yield "step_002", z
        yield "step_003", z
