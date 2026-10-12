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
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = cuda_extension.conv_transpose1d_bias_free_forward(
            x, self.conv_transpose.weight, self.stride, self.padding, self.output_padding
        )
        x = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        mask = cuda_extension.mask_not_equal_forward(x, self.mask_val)
        x = cuda_extension.masked_select_from_mask_forward(x, mask)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.conv_transpose1d_bias_free_forward(
            x, self.conv_transpose.weight, self.stride, self.padding, self.output_padding
        )
        yield "step_001", x
        x = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        yield "step_002", x
        mask = cuda_extension.mask_not_equal_forward(x, self.mask_val)
        yield "step_003", mask
        x = cuda_extension.masked_select_from_mask_forward(x, mask)
        yield "step_004", x
        yield "step_005", x
