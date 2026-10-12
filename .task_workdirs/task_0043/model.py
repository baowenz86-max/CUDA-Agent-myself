import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape, einsum_equation, clip_min, clip_max):
        super().__init__()
        self.input_shape = input_shape
        self.einsum_equation = einsum_equation
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, x):
        x = torch.view_as_real(x)
        x = torch.einsum(self.einsum_equation, x)
        x = torch.clip(x, min=self.clip_min, max=self.clip_max)
        return x


batch_size = 64
input_shape = (batch_size, 2048, 2048)  # Increased spatial dimensions
einsum_equation = "bchw->bhwc"
clip_min = 0.0
clip_max = 1.0


def get_inputs():
    return [torch.randn(*input_shape, dtype=torch.complex64)]


def get_init_inputs():
    return [input_shape, einsum_equation, clip_min, clip_max]