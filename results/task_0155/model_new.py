import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x):
        x = cuda_extension.conv3d_forward(x, self.conv3d.weight, self.conv3d.bias, self.conv3d.stride, self.conv3d.padding, self.conv3d.dilation)
        x = cuda_extension.tanh_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv3d_forward(x, self.conv3d.weight, self.conv3d.bias, self.conv3d.stride, self.conv3d.padding, self.conv3d.dilation)
        yield "step_001", x
        x = cuda_extension.tanh_forward(x)
        yield "step_002", x
        yield "step_003", x
