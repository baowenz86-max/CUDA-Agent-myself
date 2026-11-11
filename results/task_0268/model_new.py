import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, cat_dim):
        super().__init__()
        self.cat_dim = cat_dim

    def forward(self, x, y):
        x = cuda_extension.log_softmax_dim(x, self.cat_dim)
        y = cuda_extension.log_softmax_dim(y, self.cat_dim)
        return cuda_extension.cat_dim(x, y, self.cat_dim)

    def debug_forward(self, x, y):
        x = cuda_extension.log_softmax_dim(x, self.cat_dim)
        yield 'step_001', x
        y = cuda_extension.log_softmax_dim(y, self.cat_dim)
        yield 'step_002', y
        out = cuda_extension.cat_dim(x, y, self.cat_dim)
        yield 'step_003', out
