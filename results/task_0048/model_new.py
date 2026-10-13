import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU()
        self.conv_transpose3d = nn.ConvTranspose3d(
            out_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.adaptive_avg_pool2d_5d(x, 1)
        x = cuda_extension.conv3d_forward(
            x, self.conv3d.weight, self.conv3d.bias,
            self.conv3d.stride[0], self.conv3d.padding[0]
        )
        x = cuda_extension.leaky_relu_forward(x, 0.01)
        x = cuda_extension.add_bias(x, self.bias)
        x = cuda_extension.clamp_forward(x, 0.0, 1.0)
        x = cuda_extension.conv_transpose3d_forward(
            x, self.conv_transpose3d.weight, self.conv_transpose3d.bias,
            self.conv_transpose3d.stride[0], self.conv_transpose3d.padding[0],
            self.conv_transpose3d.output_padding[0]
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.adaptive_avg_pool2d_5d(x, 1)
        yield ("step_001", x)
        x = cuda_extension.conv3d_forward(
            x, self.conv3d.weight, self.conv3d.bias,
            self.conv3d.stride[0], self.conv3d.padding[0]
        )
        yield ("step_002", x)
        x = cuda_extension.leaky_relu_forward(x, 0.01)
        yield ("step_003", x)
        x = cuda_extension.add_bias(x, self.bias)
        yield ("step_004", x)
        x = cuda_extension.clamp_forward(x, 0.0, 1.0)
        yield ("step_005", x)
        x = cuda_extension.conv_transpose3d_forward(
            x, self.conv_transpose3d.weight, self.conv_transpose3d.bias,
            self.conv_transpose3d.stride[0], self.conv_transpose3d.padding[0],
            self.conv_transpose3d.output_padding[0]
        )
        yield ("step_006", x)
        yield ("step_007", x)
