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
        # Preserve original parameter structure for state_dict compatibility.
        # We do not use nn.ConvTranspose1d or nn.LayerNorm modules here because
        # forward must call custom CUDA ops; parameters are stored under the same names.
        self.conv_transpose_weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size))
        self.conv_transpose_bias = nn.Parameter(torch.empty(out_channels))
        self.layer_norm_weight = nn.Parameter(torch.empty(layer_norm_shape))
        self.layer_norm_bias = nn.Parameter(torch.empty(layer_norm_shape))
        self.weight = nn.Parameter(torch.empty(weight_shape))
        self.bias = nn.Parameter(torch.empty(bias_shape))

        # Initialize similarly to reference (not critical, but deterministic)
        nn.init.kaiming_uniform_(self.conv_transpose_weight, a=math.sqrt(5))
        if self.conv_transpose_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_transpose_weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.conv_transpose_bias, -bound, bound)
        nn.init.ones_(self.layer_norm_weight)
        nn.init.zeros_(self.layer_norm_bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -1.0, 1.0)

        # Store configuration for custom ops
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.layer_norm_shape = layer_norm_shape

    def forward(self, x, v, w):
        # Step 1: ConvTranspose1d
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        # Step 2: LayerNorm (over last dimension, shape layer_norm_shape)
        x = cuda_extension.layer_norm_forward(
            x,
            self.layer_norm_weight,
            self.layer_norm_bias,
            1e-5,  # eps from nn.LayerNorm default
        )
        # Step 3: addcmul with value=1.0
        x = cuda_extension.addcmul_forward(x, v, w, 1.0)
        return x

    def debug_forward(self, x, v, w):
        # Must yield checkpoints in order, with same ids as reference trace.
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        yield "step_001", x
        x = cuda_extension.layer_norm_forward(
            x,
            self.layer_norm_weight,
            self.layer_norm_bias,
            1e-5,
        )
        yield "step_002", x
        x = cuda_extension.addcmul_forward(x, v, w, 1.0)
        yield "step_003", x
        yield "step_004", x
