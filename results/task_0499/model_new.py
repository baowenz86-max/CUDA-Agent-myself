import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, clamp_min, clamp_max, argmin_dim):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.argmin_dim = argmin_dim

    def forward(self, x):
        x_clamped = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        x_argmin = cuda_extension.argmin_forward(x_clamped, self.argmin_dim)
        return x_argmin

    def debug_forward(self, x):
        x_clamped = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        yield "step_001", x_clamped
        x_argmin = cuda_extension.argmin_forward(x_clamped, self.argmin_dim)
        yield "step_002", x_argmin
        yield "step_003", x_argmin
