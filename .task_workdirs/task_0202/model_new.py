import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        x = cuda_extension.l2_normalize_forward(x, p=2.0, eps=1e-5, dim=1)
        x = cuda_extension.conv_transpose2d_forward(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.conv_transpose.stride, self.conv_transpose.padding, self.conv_transpose.output_padding
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.l2_normalize_forward(x, p=2.0, eps=1e-5, dim=1)
        yield 'step_001', x
        x = cuda_extension.conv_transpose2d_forward(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.conv_transpose.stride, self.conv_transpose.padding, self.conv_transpose.output_padding
        )
        yield 'step_002', x
        yield 'step_003', x
