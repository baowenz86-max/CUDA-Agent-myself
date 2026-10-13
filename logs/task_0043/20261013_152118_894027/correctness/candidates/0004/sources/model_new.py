import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape, einsum_equation, clip_min, clip_max):
        super().__init__()
        self.input_shape = input_shape
        self.einsum_equation = einsum_equation
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x):
        x = cuda_extension.view_as_real_kernel(x)
        x = cuda_extension.einsum_bchw_to_bhwc(x)
        x = cuda_extension.clip_kernel(x, self.clip_min, self.clip_max)
        return x

    def debug_forward(self, x):
        x = cuda_extension.view_as_real_kernel(x)
        yield "step_001", x
        x = cuda_extension.einsum_bchw_to_bhwc(x)
        yield "step_002", x
        x = cuda_extension.clip_kernel(x, self.clip_min, self.clip_max)
        yield "step_003", x
        yield "step_004", x
