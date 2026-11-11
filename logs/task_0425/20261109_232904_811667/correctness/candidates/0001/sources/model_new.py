import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_dim, window_size):
        super(ModelNew, self).__init__()
        self.window_size = window_size

    def forward(self, x):
        x_max = cuda_extension.max_dim_1_keepdim(x)
        x_triu = cuda_extension.triu_last_two_dims(x, diagonal=0)
        return x_max, x_triu

    def debug_forward(self, x):
        x_max = cuda_extension.max_dim_1_keepdim(x)
        yield 'step_001', x_max
        x_triu = cuda_extension.triu_last_two_dims(x, diagonal=0)
        yield 'step_002', x_triu
        yield 'step_003', (x_max, x_triu)
