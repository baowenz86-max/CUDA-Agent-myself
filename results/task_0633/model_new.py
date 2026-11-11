import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, cat_dim, argmax_dim):
        super().__init__()
        self.input_size = input_size
        self.cat_dim = cat_dim
        self.argmax_dim = argmax_dim

    def forward(self, x):
        half_size = self.input_size // 2
        cat_result = cuda_extension.cat_forward(x, half_size, self.cat_dim)
        argmax_result = cuda_extension.argmax_forward(x, self.argmax_dim)
        return cat_result, argmax_result

    def debug_forward(self, x):
        half_size = self.input_size // 2
        yield 'step_001', half_size
        x1 = x[:, :, :half_size]
        x2 = x[:, :, half_size:]
        yield 'step_002', (x1, x2)
        cat_result = cuda_extension.cat_forward(x, half_size, self.cat_dim)
        yield 'step_003', cat_result
        argmax_result = cuda_extension.argmax_forward(x, self.argmax_dim)
        yield 'step_004', argmax_result
        yield 'step_005', (cat_result, argmax_result)
