import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters, state_dict is empty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.avg_pool3d_forward(x)

    def debug_forward(self, *inputs):
        x = inputs[0]
        result = self.forward(x)
        yield 'step_001', result
