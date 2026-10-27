import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        # Initialize ConvTranspose1d parameters to match state_dict keys
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        # Store hyperparameters for forward pass
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.bias = bias

    def forward(self, x):
        # Step 1: SiLU activation
        x = cuda_extension.silu_forward(x)
        # Step 2: ConvTranspose1d
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding
        )
        return x

    def debug_forward(self, x):
        # Step 1: SiLU activation
        x = cuda_extension.silu_forward(x)
        yield 'step_001', x
        # Step 2: ConvTranspose1d
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding
        )
        yield 'step_002', x
        # Step 3: return x (same as step_002 output)
        yield 'step_003', x
