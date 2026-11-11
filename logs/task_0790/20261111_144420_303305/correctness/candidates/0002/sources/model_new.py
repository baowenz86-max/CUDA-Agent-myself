import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, channels, num_parameters):
        super().__init__()
        self.prelu = nn.PReLU(num_parameters=num_parameters)

    def forward(self, x):
        x = cuda_extension.sum_dim1_keepdim(x)
        x = cuda_extension.prelu(x, self.prelu.weight)
        return x

    def debug_forward(self, x):
        x = cuda_extension.sum_dim1_keepdim(x)
        yield 'step_001', x
        x = cuda_extension.prelu(x, self.prelu.weight)
        yield 'step_002', x
        yield 'step_003', x


def get_inputs():
    batch_size = 16
    channels = 3
    height = 2560
    width = 2560
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    channels = 3
    num_parameters = 1
    return [channels, num_parameters]
