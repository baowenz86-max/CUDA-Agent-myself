import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, power_factor: int):
        super().__init__()
        self.power_factor = power_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: x = torch.abs(x)
        x = cuda_extension.abs_forward(x)
        # Step 2: x = torch.pow(x, self.power_factor)
        x = cuda_extension.pow_scalar_forward(x, self.power_factor)
        return x

    def debug_forward(self, *inputs):
        # inputs is expected to be a single tensor x
        x = inputs[0]
        # Step 1
        abs_x = cuda_extension.abs_forward(x)
        yield "step_001", abs_x
        # Step 2
        pow_x = cuda_extension.pow_scalar_forward(abs_x, self.power_factor)
        yield "step_002", pow_x
        # Step 3: final return
        yield "step_003", pow_x
