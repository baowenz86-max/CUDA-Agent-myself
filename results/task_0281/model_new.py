import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = cuda_extension.relu_forward(x)
        x = cuda_extension.avg_pool1d_forward(x, kernel_size=3, stride=2, padding=1)
        return x

    def debug_forward(self, x):
        x = cuda_extension.relu_forward(x)
        yield 'step_001', x
        x = cuda_extension.avg_pool1d_forward(x, kernel_size=3, stride=2, padding=1)
        yield 'step_002', x
        yield 'step_003', x
