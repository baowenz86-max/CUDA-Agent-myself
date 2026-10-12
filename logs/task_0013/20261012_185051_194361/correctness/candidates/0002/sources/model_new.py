import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        # Preserve original parameter structure and state_dict keys.
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x):
        # Step 1: transposed convolution
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias if self.conv_transpose1d.bias is not None else None,
            self.conv_transpose1d.stride,
            self.conv_transpose1d.padding,
            self.conv_transpose1d.output_padding,
        )
        # Step 2: rsqrt(abs(x) + 1e-12)
        x = cuda_extension.rsqrt_eps_forward(x)
        # Step 3: softmax along last dimension
        x = cuda_extension.softmax_dim_forward(x, dim=-1)
        return x

    def debug_forward(self, x):
        # Yield each checkpoint in the same order as the reference trace.
        # Checkpoint 1: conv_transpose1d
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias if self.conv_transpose1d.bias is not None else None,
            self.conv_transpose1d.stride,
            self.conv_transpose1d.padding,
            self.conv_transpose1d.output_padding,
        )
        yield "step_001", x
        # Checkpoint 2: rsqrt(abs(x) + 1e-12)
        x = cuda_extension.rsqrt_eps_forward(x)
        yield "step_002", x
        # Checkpoint 3: softmax(dim=-1)
        x = cuda_extension.softmax_dim_forward(x, dim=-1)
        yield "step_003", x
        # Checkpoint 4: final return value
        yield "step_004", x