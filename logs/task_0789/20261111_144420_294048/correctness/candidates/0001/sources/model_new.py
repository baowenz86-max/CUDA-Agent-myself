import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.scalar_tensor = nn.Parameter(torch.scalar_tensor(1.0))

    def forward(self, x):
        x = cuda_extension.reduce_mean_dim1_keepdim(x)
        x = cuda_extension.scalar_mul(x, self.scalar_tensor.item())
        return x

    def debug_forward(self, x):
        mean_x = cuda_extension.reduce_mean_dim1_keepdim(x)
        yield ('step_001', mean_x)
        mul_x = cuda_extension.scalar_mul(mean_x, self.scalar_tensor.item())
        yield ('step_002', mul_x)
        yield ('step_003', mul_x)
        return mul_x
