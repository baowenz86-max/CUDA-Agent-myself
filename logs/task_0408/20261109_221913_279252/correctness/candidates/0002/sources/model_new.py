import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        x_max = cuda_extension.reduce_max_kernel_forward(x, self.dim)
        x_prod = cuda_extension.reduce_prod_kernel_forward(x, self.dim)
        return x_max, x_prod

    def debug_forward(self, x: torch.Tensor):
        x_max = cuda_extension.reduce_max_kernel_forward(x, self.dim)
        yield "step_001", x_max
        x_prod = cuda_extension.reduce_prod_kernel_forward(x, self.dim)
        yield "step_002", x_prod
        yield "step_003", (x_max, x_prod)


batch_size = 512
channels = 3
height = 2048
width = 128


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]


def get_init_inputs():
    return [2]
