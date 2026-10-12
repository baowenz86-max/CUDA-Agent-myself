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
        x = torch.ops.cuda_extension.conv_transpose_forward(x, self.conv_transpose.weight, self.conv_transpose.bias,
                                                             self.conv_transpose.stride, self.conv_transpose.padding,
                                                             self.conv_transpose.output_padding)
        x = torch.ops.cuda_extension.clamp_forward(x, 0.0, 1.0)
        x = torch.ops.cuda_extension.leaky_relu_forward(x, 0.01)
        x = torch.ops.cuda_extension.triu_forward(x, 0)
        x = torch.ops.cuda_extension.add_forward(x, self.bias)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = torch.ops.cuda_extension.conv_transpose_forward(x, self.conv_transpose.weight, self.conv_transpose.bias,
                                                             self.conv_transpose.stride, self.conv_transpose.padding,
                                                             self.conv_transpose.output_padding)
        yield "step_001", x
        x = torch.ops.cuda_extension.clamp_forward(x, 0.0, 1.0)
        yield "step_002", x
        x = torch.ops.cuda_extension.leaky_relu_forward(x, 0.01)
        yield "step_003", x
        x = torch.ops.cuda_extension.triu_forward(x, 0)
        yield "step_004", x
        x = torch.ops.cuda_extension.add_forward(x, self.bias)
        yield "step_005", x
        yield "step_006", x
