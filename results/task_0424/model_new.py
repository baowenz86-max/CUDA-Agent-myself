import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_parameters, num_channels, kernel_size, bias=True):
        super().__init__()
        # Match reference: nn.PReLU(num_parameters=num_parameters)
        self.prelu = nn.PReLU(num_parameters=num_parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.prelu_forward(x, self.prelu.weight)

    def debug_forward(self, *inputs):
        x = inputs[0]
        out = cuda_extension.prelu_forward(x, self.prelu.weight)
        yield 'step_001', out
