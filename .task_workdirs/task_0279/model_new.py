import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, intermediate_channels, groups, p
    ):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.conv1d = nn.Conv1d(out_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, groups=groups)
        self.p = p

    def forward(self, x):
        x = cuda_extension.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            bias=self.conv_transpose1d.bias,
            stride=self.conv_transpose1d.stride,
            padding=self.conv_transpose1d.padding,
            output_padding=self.conv_transpose1d.output_padding,
        )
        x = cuda_extension.normalize(x, p=self.p, dim=1)
        x = cuda_extension.conv1d(x, self.conv1d.weight, bias=self.conv1d.bias, stride=1, padding=1, groups=self.conv1d.groups)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv_transpose1d(
            x,
            self.conv_transpose1d.weight,
            bias=self.conv_transpose1d.bias,
            stride=self.conv_transpose1d.stride,
            padding=self.conv_transpose1d.padding,
            output_padding=self.conv_transpose1d.output_padding,
        )
        yield "step_001", x
        x = cuda_extension.normalize(x, p=self.p, dim=1)
        yield "step_002", x
        x = cuda_extension.conv1d(x, self.conv1d.weight, bias=self.conv1d.bias, stride=1, padding=1, groups=self.conv1d.groups)
        yield "step_003", x
        yield "step_004", x
