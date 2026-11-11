import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, input_dim):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.input_dim = input_dim

    def forward(self, x):
        assert x.dim() == self.input_dim, f"Expected input dim to be {self.input_dim}, but got {x.dim()}"
        return cuda_extension.conv2d_relu(
            x,
            self.conv2d.weight,
            self.conv2d.bias,
            self.conv2d.stride[0],
            self.conv2d.padding[0],
            self.conv2d.dilation[0],
            self.conv2d.groups,
        )

    def debug_forward(self, x):
        out = self.forward(x)
        return [(None, out)]
