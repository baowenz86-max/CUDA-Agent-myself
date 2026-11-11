import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, channel_num, negative_slope, max_value):
        super(ModelNew, self).__init__()
        self.prelu = nn.PReLU(channel_num, init=negative_slope)
        self.max_value = max_value

    def forward(self, x):
        x = cuda_extension.custom_clamp_max(x, self.max_value)
        x = cuda_extension.custom_prelu(x, self.prelu.weight)
        return x

    def debug_forward(self, x):
        max_value_tensor = torch.full_like(x, self.max_value)
        yield 'step_001', max_value_tensor
        clamped = cuda_extension.custom_clamp_max(x, self.max_value)
        yield 'step_002', clamped
        prelu_out = cuda_extension.custom_prelu(clamped, self.prelu.weight)
        yield 'step_003', prelu_out
        yield 'step_004', prelu_out


batch_size = 512
channels = 64
height = 64
width = 64


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [64, 0.25, 2.0]
