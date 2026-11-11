import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, padding_size, padding_value):
        super().__init__()
        self.padding_size = padding_size
        self.padding_value = padding_value
        # Pack padding tuple (left, right) into integer: left*10+right
        # Here (1,2) -> 12
        pad_left, pad_right = self.padding_size
        self.padding_int = pad_left * 10 + pad_right

    def forward(self, x):
        x = cuda_extension.custom_softmax_dim1(x)
        x = cuda_extension.custom_constant_pad2d(x, self.padding_int, self.padding_value)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_softmax_dim1(x)
        yield "step_001", x
        x = cuda_extension.custom_constant_pad2d(x, self.padding_int, self.padding_value)
        yield "step_002", x
        yield "step_003", x
