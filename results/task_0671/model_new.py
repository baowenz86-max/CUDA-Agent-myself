import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reference: x = torch.abs(x)
        x = cuda_extension.custom_abs(x)
        # Reference: x = torch.floor(x)
        x = cuda_extension.custom_floor(x)
        return x

    def debug_forward(self, *inputs):
        # inputs is expected to contain a single tensor x
        x = inputs[0]
        x_abs = cuda_extension.custom_abs(x)
        yield ("step_001", x_abs)
        x_floor = cuda_extension.custom_floor(x_abs)
        yield ("step_002", x_floor)
        yield ("step_003", x_floor)
