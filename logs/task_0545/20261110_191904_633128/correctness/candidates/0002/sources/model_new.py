import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, height, width):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.copy_to_cuda(x)
        max_result = cuda_extension.max_step(x)
        squeezed = cuda_extension.squeeze_step(max_result.values)
        output = cuda_extension.view_as_complex_step(squeezed)
        return output

    def debug_forward(self, x):
        x = cuda_extension.copy_to_cuda(x)
        max_result = cuda_extension.max_step(x)
        yield "step_001", max_result
        squeezed = cuda_extension.squeeze_step(max_result.values)
        yield "step_002", squeezed
        output = cuda_extension.view_as_complex_step(squeezed)
        yield "step_003", output
