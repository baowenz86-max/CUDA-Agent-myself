import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, padding_size):
        super().__init__()
        # Store padding parameters; kernel_size and in_channels are unused but kept for signature compatibility
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding_size = padding_size
        # padding_size is expected to be a tuple (left, right, top, bottom) as in nn.ReflectionPad2d
        if isinstance(padding_size, int):
            self.pad_left = self.pad_right = self.pad_top = self.pad_bottom = padding_size
        else:
            self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = padding_size

    def forward(self, x):
        # Call custom op with padding parameters
        return cuda_extension.custom_ops_forward(
            x, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom
        )

    def debug_forward(self, x):
        # Step 1: Reflection pad (custom op)
        padded = cuda_extension.reflection_pad2d_forward(
            x, self.pad_left, self.pad_right, self.pad_top, self.pad_bottom
        )
        yield 'step_001', padded
        # Step 2: Greater equal zero (custom op)
        ge = cuda_extension.ge_forward(padded, 0)
        yield 'step_002', ge
        # Step 3: Remainder (custom op)
        rem = cuda_extension.remainder_forward(ge, 2)
        yield 'step_003', rem
        # Step 4: Return (same as rem)
        yield 'step_004', rem
        return rem