import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=True
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x):
        conv_out = cuda_extension.conv3d_forward(
            x,
            self.conv3d.weight,
            self.conv3d.bias if self.conv3d.bias is not None else None,
            self.conv3d.stride,
            self.conv3d.padding,
            self.conv3d.dilation,
        )
        return cuda_extension.softplus_forward(conv_out)

    def debug_forward(self, x):
        conv_out = cuda_extension.conv3d_forward(
            x,
            self.conv3d.weight,
            self.conv3d.bias if self.conv3d.bias is not None else None,
            self.conv3d.stride,
            self.conv3d.padding,
            self.conv3d.dilation,
        )
        out = cuda_extension.softplus_forward(conv_out)
        yield 'step_001', out
