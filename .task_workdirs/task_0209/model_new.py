import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        # Keep the same submodule structure and parameter names as the reference model
        self.conv_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Step 1: ConvTranspose1d with explicit bias (self.bias, not module bias)
        x = cuda_extension.conv_transpose1d_custom(
            x,
            self.conv_transpose.weight,
            self.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0]
        )
        # Step 2: addcmul(x, x, x) = x + x * x
        x = cuda_extension.addcmul_custom(x, x, x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        # Step 1
        x = cuda_extension.conv_transpose1d_custom(
            x,
            self.conv_transpose.weight,
            self.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0]
        )
        yield "step_001", x
        # Step 2
        x = cuda_extension.addcmul_custom(x, x, x)
        yield "step_002", x
