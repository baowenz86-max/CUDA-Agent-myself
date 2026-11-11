import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp

    def forward(self, x):
        x = x.cuda()
        x = cuda_extension.softmax_dim1_forward(x)
        x, _ = cuda_extension.sort_descending_dim1_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0].cuda()
        x = cuda_extension.softmax_dim1_forward(x)
        yield 'step_001', x
        x, indices = cuda_extension.sort_descending_dim1_forward(x)
        yield 'step_002', (x, indices)
        yield 'step_003', x
