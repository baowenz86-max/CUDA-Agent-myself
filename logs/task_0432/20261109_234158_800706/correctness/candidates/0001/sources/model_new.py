import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_dim, num_classes, dim):
        super(ModelNew, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=dim)
        self.num_classes = num_classes
        self.dim = dim

    def forward(self, x):
        x = cuda_extension.max_pool2d_forward(x, self.dim)
        _, indices = cuda_extension.max_dim1_indices(x)
        x = cuda_extension.one_hot_from_indices(indices, self.num_classes)
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.max_pool2d_forward(x, self.dim)
        yield 'step_001', x1
        _, indices = cuda_extension.max_dim1_indices(x1)
        yield 'step_002', (_, indices)
        x3 = cuda_extension.one_hot_from_indices(indices, self.num_classes)
        yield 'step_003', x3
        yield 'step_004', x3
