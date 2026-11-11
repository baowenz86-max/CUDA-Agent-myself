import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_shape, mask_value):
        super().__init__()
        self.mask_value = mask_value

    def forward(self, x):
        mask = torch.isnan(x)
        x = torch.masked_fill(x, mask, self.mask_value)
        return x


batch_size = 2048  # Increased to increase runtime
height = 512
width = 256
input_shape = (batch_size, height, width)
mask_value = 0.0


def get_inputs():
    # Generate input with shape (batch_size, height, width) containing NaNs
    x = torch.randn(batch_size, height, width, dtype=torch.float32)
    x[0, 0, 0] = float("nan")  # Add NaN to mimic original input's structure
    x[1, 1, 1] = float("nan")
    return [x]


def get_init_inputs():
    return [input_shape, mask_value]