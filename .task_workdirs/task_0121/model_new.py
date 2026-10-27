import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # 保留参数结构，与参考 state_dict 键完全一致
        self.padding = nn.ConstantPad1d(padding, 0)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # 保存参数供自定义算子使用
        self.padding_left = padding[0]
        self.padding_right = padding[1]
        self.stride = stride
        self.output_padding = output_padding

    def forward(self, x):
        x = cuda_extension.custom_pad1d_forward(
            x, self.padding_left, self.padding_right
        )
        x = cuda_extension.custom_conv_transpose2d_forward(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.stride, self.output_padding
        )
        x = cuda_extension.custom_add_bias_forward(
            x, self.bias
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_pad1d_forward(
            x, self.padding_left, self.padding_right
        )
        yield 'step_001', x
        x = cuda_extension.custom_conv_transpose2d_forward(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.stride, self.output_padding
        )
        yield 'step_002', x
        x = cuda_extension.custom_add_bias_forward(
            x, self.bias
        )
        yield 'step_003', x
        yield 'step_004', x
