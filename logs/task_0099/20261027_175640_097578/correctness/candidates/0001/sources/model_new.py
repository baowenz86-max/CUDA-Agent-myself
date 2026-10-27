import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        # 使用与参考相同的模块结构以保持 state_dict 兼容，但不会在 forward 中调用该模块
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        x = cuda_extension.custom_sqrt(x)
        x = cuda_extension.custom_round(x)
        x = cuda_extension.custom_conv_transpose2d(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.conv_transpose.stride[0], self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0]
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_sqrt(x)
        yield 'step_001', x
        x = cuda_extension.custom_round(x)
        yield 'step_002', x
        x = cuda_extension.custom_conv_transpose2d(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.conv_transpose.stride[0], self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0]
        )
        yield 'step_003', x
        yield 'step_004', x
