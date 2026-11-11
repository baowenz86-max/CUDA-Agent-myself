import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        return cuda_extension.flip_cat_forward(self.dim, x1, x2)

    def debug_forward(self, x1, x2):
        flipped1 = cuda_extension.flip_forward(self.dim, x1)
        yield "step_001", flipped1
        flipped2 = cuda_extension.flip_forward(self.dim, x2)
        yield "step_002", flipped2
        result = cuda_extension.cat_forward(self.dim, flipped1, flipped2)
        yield "step_003", result
