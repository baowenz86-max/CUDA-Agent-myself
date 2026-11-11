import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x):
        return cuda_extension.fused_abs_sign_forward(x)

    def debug_forward(self, x):
        step_001 = cuda_extension.fused_abs_sign_forward(x)
        yield 'step_001', step_001
        step_002 = cuda_extension.fused_abs_sign_forward(step_001)
        yield 'step_002', step_002
        yield 'step_003', step_002
