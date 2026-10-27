import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = cuda_extension.clamp_forward(x, 0.0, 1.0)
        x = cuda_extension.add_scalar_forward(x, self.scale_factor)
        return x

    def debug_forward(self, *inputs):
        # inputs 仅包含 x，不包含 self.scale_factor（参考 forward 参数一致）
        x = inputs[0]
        x = cuda_extension.clamp_forward(x, 0.0, 1.0)
        yield "step_001", x
        x = cuda_extension.add_scalar_forward(x, self.scale_factor)
        yield "step_002", x
        yield "step_003", x
