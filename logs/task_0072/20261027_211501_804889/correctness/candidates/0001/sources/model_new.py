import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        dim_tensor = torch.tensor(self.input_size, dtype=torch.int64, device=x.device)
        x = cuda_extension.mean_dim(x, dim_tensor)
        x = cuda_extension.gelu(x)
        return x

    def debug_forward(self, x):
        dim_tensor = torch.tensor(self.input_size, dtype=torch.int64, device=x.device)
        x_mean = cuda_extension.mean_dim(x, dim_tensor)
        yield 'step_001', x_mean
        x_gelu = cuda_extension.gelu(x_mean)
        yield 'step_002', x_gelu
        yield 'step_003', x_gelu
