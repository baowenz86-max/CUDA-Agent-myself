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
        bias,
        clamp_min,
        clamp_max,
        mask_val,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.mask_val = mask_val
        self._kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self._stride = stride if isinstance(stride, tuple) else (stride,)
        self._padding = padding if isinstance(padding, tuple) else (padding,)
        self._output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding,)
        self._bias = bias

    def forward(self, x):
        # x shape: (N, C_in, L_in)
        x = cuda_extension.conv_transpose1d_bias_free_forward(
            x,
            self.conv_transpose.weight,
            self._stride[0],
            self._padding[0],
            self._output_padding[0],
            self._kernel_size[0],
        )
        x = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        mask = cuda_extension.mask_not_equal_forward(x, self.mask_val)
        x = cuda_extension.masked_select_from_mask_forward(x, mask)
        return x

    def debug_forward(self, x):
        # Follow the same steps and yield checkpoints
        x = cuda_extension.conv_transpose1d_bias_free_forward(
            x,
            self.conv_transpose.weight,
            self._stride[0],
            self._padding[0],
            self._output_padding[0],
            self._kernel_size[0],
        )
        yield "step_001", x

        x = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        yield "step_002", x

        mask = cuda_extension.mask_not_equal_forward(x, self.mask_val)
        yield "step_003", mask

        x = cuda_extension.masked_select_from_mask_forward(x, mask)
        yield "step_004", x

        yield "step_005", x
