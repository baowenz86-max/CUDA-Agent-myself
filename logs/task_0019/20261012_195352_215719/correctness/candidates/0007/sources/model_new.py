import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # 调用自定义 conv_transpose 算子
        x = torch.ops.cuda_extension.conv_transpose2d_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],  # stride_h
            self.conv_transpose.stride[1],  # stride_w
            self.conv_transpose.padding[0],  # padding_h
            self.conv_transpose.padding[1],  # padding_w
            self.conv_transpose.output_padding[0],  # output_padding_h
            self.conv_transpose.output_padding[1]   # output_padding_w
        )
        # clamp [0,1]
        x = torch.ops.cuda_extension.clamp_forward(x, 0.0, 1.0)
        # leaky_relu (默认 negative_slope=0.01)
        x = torch.ops.cuda_extension.leaky_relu_forward(x, 0.01)
        # triu (diagonal=0)
        x = torch.ops.cuda_extension.triu_forward(x, 0)
        # add bias (self.bias shape [out_channels,1,1], 广播到 [N,C,H,W])
        x = torch.ops.cuda_extension.add_forward(x, self.bias)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        # checkpoint step_001: conv_transpose
        x = torch.ops.cuda_extension.conv_transpose2d_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.stride[1],
            self.conv_transpose.padding[0],
            self.conv_transpose.padding[1],
            self.conv_transpose.output_padding[0],
            self.conv_transpose.output_padding[1]
        )
        yield ("step_001", x)

        # checkpoint step_002: clamp
        x = torch.ops.cuda_extension.clamp_forward(x, 0.0, 1.0)
        yield ("step_002", x)

        # checkpoint step_003: leaky_relu
        x = torch.ops.cuda_extension.leaky_relu_forward(x, 0.01)
        yield ("step_003", x)

        # checkpoint step_004: triu
        x = torch.ops.cuda_extension.triu_forward(x, 0)
        yield ("step_004", x)

        # checkpoint step_005: add bias
        x = torch.ops.cuda_extension.add_forward(x, self.bias)
        yield ("step_005", x)

        # checkpoint step_006: return x (最终返回)
        yield ("step_006", x)
