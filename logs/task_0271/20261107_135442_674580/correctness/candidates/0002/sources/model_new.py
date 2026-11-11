import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = tuple(output_size)

    def forward(self, x):
        return cuda_extension.adaptive_avg_pool3d_forward(x, self.output_size)

    def debug_forward(self, x):
        result = cuda_extension.adaptive_avg_pool3d_forward(x, self.output_size)
        yield 'step_001', result
