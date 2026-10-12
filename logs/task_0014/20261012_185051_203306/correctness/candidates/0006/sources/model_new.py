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
        # Must match state_dict key: conv_transpose.weight
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

    def forward(self, x):
        # Step 1: conv transpose with no bias (bias=False)
        x = cuda_extension.conv_transpose1d_bias_free_forward(x, self.conv_transpose.weight)
        # Step 2: clamp
        x = cuda_extension.clamp_forward(x, self.clamp_min, self.clamp_max)
        # Step 3+4: mask and masked_select
        x = cuda_extension.masked_select_forward(x, self.mask_val)
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.conv_transpose1d_bias_free_forward(x, self.conv_transpose.weight)
        yield ("step_001", x1)
        x2 = cuda_extension.clamp_forward(x1, self.clamp_min, self.clamp_max)
        yield ("step_002", x2)
        mask = cuda_extension.mask_not_equal_forward(x2, self.mask_val)
        yield ("step_003", mask)
        x3 = cuda_extension.masked_select_from_mask_forward(x2, mask)
        yield ("step_004", x3)
        yield ("step_005", x3)
