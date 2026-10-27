import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, num_features):
        super().__init__()
        # 保持 state_dict 键与参考一致
        self.conv_transpose_1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bn_2d = nn.BatchNorm2d(num_features)

    def forward(self, x):
        x = self.conv_transpose1d_forward(x)
        x = self.unsqueeze_forward(x, 2)
        x = self.batch_norm2d_forward(x)
        x = self.squeeze_forward(x, 2)
        return x

    def conv_transpose1d_forward(self, x):
        # 自定义 CUDA 算子：转置卷积，参数来自模块
        return cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose_1d.weight,
            self.conv_transpose_1d.bias,
            self.conv_transpose_1d.stride[0],
            self.conv_transpose_1d.padding[0],
            self.conv_transpose_1d.output_padding[0]
        )

    def unsqueeze_forward(self, x, dim):
        return cuda_extension.unsqueeze_forward(x, dim)

    def batch_norm2d_forward(self, x):
        return cuda_extension.batch_norm2d_forward(
            x,
            self.bn_2d.weight,
            self.bn_2d.bias,
            self.bn_2d.running_mean,
            self.bn_2d.running_var,
            self.bn_2d.eps
        )

    def squeeze_forward(self, x, dim):
        return cuda_extension.squeeze_forward(x, dim)

    def debug_forward(self, *inputs):
        x = inputs[0]
        yield 'step_001', x
        x = self.conv_transpose1d_forward(x)
        yield 'step_002', x
        x = self.unsqueeze_forward(x, 2)
        yield 'step_003', x
        x = self.batch_norm2d_forward(x)
        yield 'step_004', x
        x = self.squeeze_forward(x, 2)
        yield 'step_005', x
