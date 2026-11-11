import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # No parameters or submodules needed; hardswish is stateless.
        # Preserve signature: in_features is unused but matches reference.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.hardswish_forward(x)

    def debug_forward(self, *inputs):
        # Single input tensor x
        x = inputs[0]
        y = cuda_extension.hardswish_forward(x)
        yield 'step_001', y