import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        # Keep original submodule/parameter structure for state_dict compatibility.
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Custom ops only; no torch operators.
        x = torch.ops.cuda_extension.conv2d_forward(x, self.conv2d.weight, self.conv2d.bias)
        x = torch.ops.cuda_extension.triu_forward(x)
        x = torch.ops.cuda_extension.bitwise_xor_forward(x, self.bias)
        return torch.ops.cuda_extension.cast_float_forward(x)

    def debug_forward(self, x):
        # Yield checkpoints in order, matching reference trace.
        x = torch.ops.cuda_extension.conv2d_forward(x, self.conv2d.weight, self.conv2d.bias)
        yield "step_001", x
        x = torch.ops.cuda_extension.triu_forward(x)
        yield "step_002", x
        x = torch.ops.cuda_extension.bitwise_xor_forward(x, self.bias)
        yield "step_003", x
        x = torch.ops.cuda_extension.cast_float_forward(x)
        yield "step_004", x
