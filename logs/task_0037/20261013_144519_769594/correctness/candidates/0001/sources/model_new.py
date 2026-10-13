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
        # Convert to one-hot encoding
        x = cuda_extension.one_hot_forward(x, self.num_classes)
        # Cast to float
        x = cuda_extension.float_forward(x)
        # Pad the tensor
        x = cuda_extension.pad_forward(x, self.padding_size, self.padding_value)
        # Apply softmax
        x = cuda_extension.softmax_forward(x, self.depth)
        return x

    def debug_forward(self, x):
        # Same as forward but yield checkpoints
        x = cuda_extension.one_hot_forward(x, self.num_classes)
        yield 'step_001', x
        x = cuda_extension.float_forward(x)
        yield 'step_002', x
        x = cuda_extension.pad_forward(x, self.padding_size, self.padding_value)
        yield 'step_003', x
        x = cuda_extension.softmax_forward(x, self.depth)
        yield 'step_004', x
        yield 'step_005', x
