import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape1, input_shape2):
        super().__init__()
        self.input_shape1 = input_shape1
        self.input_shape2 = input_shape2

    def forward(self, x1, x2):
        x1 = cuda_extension.cat_dim1(x1, x2)
        x1 = cuda_extension.clone_tensor(x1)
        return x1

    def debug_forward(self, x1, x2):
        yield 'step_001', cuda_extension.cat_dim1(x1, x2)
        x1 = cuda_extension.cat_dim1(x1, x2)
        yield 'step_002', cuda_extension.clone_tensor(x1)
        x1 = cuda_extension.clone_tensor(x1)
        yield 'step_003', x1
