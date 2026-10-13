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
        # one_hot_float now accepts CPU or CUDA input, handling device accordingly.
        x = cuda_extension.one_hot_float(x, self.num_classes)
        x = cuda_extension.pad_constant(x, self.padding_size, self.padding_size, self.padding_size, self.padding_size, self.padding_value)
        x = cuda_extension.softmax_dim(x, self.depth)
        return x

    def debug_forward(self, x):
        onehot = cuda_extension.one_hot_int64(x, self.num_classes)
        yield 'step_001', onehot
        x_f = cuda_extension.cast_float32(onehot)
        yield 'step_002', x_f
        padded = cuda_extension.pad_constant(x_f, self.padding_size, self.padding_size, self.padding_size, self.padding_size, self.padding_value)
        yield 'step_003', padded
        result = cuda_extension.softmax_dim(padded, self.depth)
        yield 'step_004', result
        yield 'step_005', result
