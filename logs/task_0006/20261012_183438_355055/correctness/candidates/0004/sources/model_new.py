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
