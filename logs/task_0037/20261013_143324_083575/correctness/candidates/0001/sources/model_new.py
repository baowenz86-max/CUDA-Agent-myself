import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_classes, depth, padding_size, padding_value):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.padding_size = padding_size
        self.padding_value = padding_value

    def forward(self, x):
        # Step 1: one_hot + float cast in one custom op
        # Input x is int64 CPU, but custom op will move to CUDA and produce float32.
        x = cuda_extension.one_hot_float(x, self.num_classes)
        # Step 2: constant padding on height and width dims
        x = cuda_extension.pad_constant(x, self.padding_size, self.padding_size, self.padding_size, self.padding_size, self.padding_value)
        # Step 3: softmax over depth dimension (last dimension)
        x = cuda_extension.softmax_dim(x, self.depth)
        return x

    def debug_forward(self, x):
        # Must yield checkpoint IDs and values in exact order as reference_steps
        # step_001: one_hot result (int64 tensor on cuda)
        onehot = cuda_extension.one_hot_int64(x, self.num_classes)
        yield 'step_001', onehot
        # step_002: float cast
        x_f = cuda_extension.cast_float32(onehot)
        yield 'step_002', x_f
        # step_003: pad
        padded = cuda_extension.pad_constant(x_f, self.padding_size, self.padding_size, self.padding_size, self.padding_size, self.padding_value)
        yield 'step_003', padded
        # step_004: softmax
        result = cuda_extension.softmax_dim(padded, self.depth)
        yield 'step_004', result
        # step_005: return result (same as step_004)
        yield 'step_005', result
