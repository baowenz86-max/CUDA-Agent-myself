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
        x = cuda_extension.one_hot_forward(x, self.num_classes)
        x = cuda_extension.pad_forward(x, self.padding_size, self.padding_value)
        x = cuda_extension.softmax_forward(x, self.depth)
        return x

    def debug_forward(self, x):
        x_one_hot = cuda_extension.one_hot_forward(x, self.num_classes)
        yield "step_001", x_one_hot
        x_float = x_one_hot.float()
        yield "step_002", x_float
        x_pad = cuda_extension.pad_forward(x_float, self.padding_size, self.padding_value)
        yield "step_003", x_pad
        x_out = cuda_extension.softmax_forward(x_pad, self.depth)
        yield "step_004", x_out
        yield "step_005", x_out
