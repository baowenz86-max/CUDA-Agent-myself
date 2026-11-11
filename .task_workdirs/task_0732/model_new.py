import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_dim, mid_dim):
        super().__init__()
        self.mid_dim = mid_dim

    def forward(self, x):
        x = cuda_extension.acos_stack(x)
        x = cuda_extension.cat_unbind_channels(x)
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.acos_stack(x)
        yield "step_001", x1
        x2 = cuda_extension.cat_unbind_channels(x1)
        yield "step_002", x2
        yield "step_003", x2


batch_size = 16
in_channels = 64
height = 512
width = 256
mid_dim = 64


def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width).clamp(-1, 1)
    return [x]


def get_init_inputs():
    return [in_channels, mid_dim]
