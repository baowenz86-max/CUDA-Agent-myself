import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, index_size, dim, src_size):
        super().__init__()
        self.input_size = input_size
        self.index_size = index_size
        self.dim = dim
        self.src_size = src_size

    def forward(self, input_tensor, index, src):
        mean_value = cuda_extension.custom_mean_dim(input_tensor, self.dim)
        scattered_tensor = cuda_extension.custom_scatter_dim(input_tensor, self.dim, index, src)
        return mean_value, scattered_tensor

    def debug_forward(self, input_tensor, index, src):
        mean_value = cuda_extension.custom_mean_dim(input_tensor, self.dim)
        yield 'step_001', mean_value
        scattered_tensor = cuda_extension.custom_scatter_dim(input_tensor, self.dim, index, src)
        yield 'step_002', scattered_tensor
        yield 'step_003', (mean_value, scattered_tensor)
