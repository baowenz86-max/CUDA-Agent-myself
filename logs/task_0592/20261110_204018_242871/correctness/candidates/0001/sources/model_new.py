import torch

import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, cat_dim):
        super().__init__()
        self.cat_dim = cat_dim

    def forward(self, x1, x2):
        x = cuda_extension.cat_forward(x1, x2, int(self.cat_dim))
        x = cuda_extension.ceil_forward(x)
        return x

    def debug_forward(self, x1, x2):
        x = cuda_extension.cat_forward(x1, x2, int(self.cat_dim))
        yield 'step_001', x
        x = cuda_extension.ceil_forward(x)
        yield 'step_002', x
        yield 'step_003', x
