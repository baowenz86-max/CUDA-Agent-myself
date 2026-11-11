import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def forward(self, x):
        x = cuda_extension.elu_forward(x)
        x = cuda_extension.mean_all_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.elu_forward(x)
        yield 'step_001', x
        x = cuda_extension.mean_all_forward(x)
        yield 'step_002', x
        yield 'step_003', x
