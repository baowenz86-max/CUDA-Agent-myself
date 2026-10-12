import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=bias,
        )

    def forward(self, x):
        conv_out = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        rsqrt_out = cuda_extension.rsqrt_eps_forward(conv_out, 1e-12)
        softmax_out = cuda_extension.softmax_lastdim_forward(rsqrt_out)
        return softmax_out

    def debug_forward(self, x):
        conv_out = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        yield ("step_001", conv_out)
        rsqrt_out = cuda_extension.rsqrt_eps_forward(conv_out, 1e-12)
        yield ("step_002", rsqrt_out)
        softmax_out = cuda_extension.softmax_lastdim_forward(rsqrt_out)
        yield ("step_003", softmax_out)
        yield ("step_004", softmax_out)
