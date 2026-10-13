import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        layer_norm_shape,
        weight_shape,
        bias_shape,
    ):
        super().__init__()
        # Store configuration and create the same submodules/parameters as Model.
        # The submodules are only used as parameter containers; their forward is never called.
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.layer_norm = nn.LayerNorm(layer_norm_shape)
        self.weight = nn.Parameter(torch.randn(weight_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, v, w):
        # Step 001: ConvTranspose1d with bias
        x = cuda_extension.conv_transpose_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
            output_padding=self.conv_transpose.output_padding[0],
        )
        # Step 002: LayerNorm over the last dimension (layer_norm_shape)
        x = cuda_extension.layer_norm_forward(
            x,
            self.layer_norm.weight,
            self.layer_norm.bias,
            eps=self.layer_norm.eps,
        )
        # Step 003: x = x + v * w
        x = cuda_extension.addcmul_forward(x, v, w)
        return x

    def debug_forward(self, *inputs):
        # Unpack exactly like forward
        x, v, w = inputs
        # Step 001: ConvTranspose1d with bias
        x = cuda_extension.conv_transpose_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
            output_padding=self.conv_transpose.output_padding[0],
        )
        yield "step_001", x

        # Step 002: LayerNorm over the last dimension (layer_norm_shape)
        x = cuda_extension.layer_norm_forward(
            x,
            self.layer_norm.weight,
            self.layer_norm.bias,
            eps=self.layer_norm.eps,
        )
        yield "step_002", x

        # Step 003: x = x + v * w
        x = cuda_extension.addcmul_forward(x, v, w)
        yield "step_003", x

        # Step 004: return x
        yield "step_004", x
