import torch
import torch.nn as nn
import torch.nn.functional as F
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # 保持与参考完全相同的 state_dict 键
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        x = cuda_extension.mul_bias_forward(x, self.bias)
        return x

    def debug_forward(self, x):
        step_001 = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        yield ('step_001', step_001)
        step_002 = cuda_extension.mul_bias_forward(step_001, self.bias)
        yield ('step_002', step_002)
        yield ('step_003', step_002)
