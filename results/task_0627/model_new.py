import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, pad_size, channels):
        super().__init__()
        self.pad_size = pad_size
        self.channels = channels

    def forward(self, x):
        # pad_size is (left, right, top, bottom)
        left, right, top, bottom = self.pad_size
        return cuda_extension.pad_sum_channels(
            x, left, right, top, bottom
        )

    def debug_forward(self, x):
        left, right, top, bottom = self.pad_size
        # Call custom op once to produce final output
        out = cuda_extension.pad_sum_channels(x, left, right, top, bottom)
        # We need intermediate padded tensor for checkpoint step_001
        # Since custom op produces only final sum, we need a separate op for padding only
        # Define custom op pad_only to get padded tensor, then sum via custom op
        padded = cuda_extension.pad2d_custom(x, left, right, top, bottom)
        yield ('step_001', padded)
        # Sum channels using custom op (could be a separate sum_channels_custom)
        summed = cuda_extension.sum_channels_custom(padded, keepdim=True)
        yield ('step_002', summed)
        yield ('step_003', summed)
