import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters, but keep for potential state_dict compatibility
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = cuda_extension.avgpool2d_forward(x, 2, 2, 0)
        x = cuda_extension.relu_forward(x, False)
        return x

    def debug_forward(self, x):
        x = cuda_extension.avgpool2d_forward(x, 2, 2, 0)
        yield 'step_001', x
        x = cuda_extension.relu_forward(x, False)
        yield 'step_002', x
        yield 'step_003', x
