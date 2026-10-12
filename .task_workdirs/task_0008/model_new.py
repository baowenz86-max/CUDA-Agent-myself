import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias_shape):
        super().__init__()
        # Store parameters to match reference state_dict keys
        self.padding = padding  # expected tuple of ints
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # ConstantPad2d with 2-tuple padding (symmetry): expand to 4-tuple (left, right, top, bottom)
        if len(self.padding) == 2:
            pad_left, pad_right, pad_top, pad_bottom = self.padding[0], self.padding[0], self.padding[1], self.padding[1]
        else:
            pad_left, pad_right, pad_top, pad_bottom = self.padding
        x = cuda_extension.pad_constant_2d(x, [pad_left, pad_right, pad_top, pad_bottom], self.padding[0])
        x = cuda_extension.rot90_k1(x)
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias)
        mask = cuda_extension.isinf(x)
        x = cuda_extension.masked_fill_zero(x, mask)
        x = cuda_extension.add_bias(x, self.bias)
        return x

    def debug_forward(self, x):
        if len(self.padding) == 2:
            pad_left, pad_right, pad_top, pad_bottom = self.padding[0], self.padding[0], self.padding[1], self.padding[1]
        else:
            pad_left, pad_right, pad_top, pad_bottom = self.padding
        x = cuda_extension.pad_constant_2d(x, [pad_left, pad_right, pad_top, pad_bottom], self.padding[0])
        yield ("step_001", x)
        x = cuda_extension.rot90_k1(x)
        yield ("step_002", x)
        x = cuda_extension.conv2d_forward(x, self.conv.weight, self.conv.bias)
        yield ("step_003", x)
        mask = cuda_extension.isinf(x)
        yield ("step_004", mask)
        x = cuda_extension.masked_fill_zero(x, mask)
        yield ("step_005", x)
        x = cuda_extension.add_bias(x, self.bias)
        yield ("step_006", x)
        yield ("step_007", x)
