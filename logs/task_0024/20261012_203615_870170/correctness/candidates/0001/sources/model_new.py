import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, threshold):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.threshold = threshold

    def forward(self, x, z):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        median_value = cuda_extension.median_forward(x)
        x = cuda_extension.ne_forward(x, median_value)
        z = cuda_extension.ne_forward(z, self.threshold)
        return x, z

    def debug_forward(self, *inputs):
        x, z = inputs
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose1d.weight,
            self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0],
        )
        yield 'step_001', x
        median_value = cuda_extension.median_forward(x)
        yield 'step_002', median_value
        x = cuda_extension.ne_forward(x, median_value)
        yield 'step_003', x
        z = cuda_extension.ne_forward(z, self.threshold)
        yield 'step_004', z
        yield 'step_005', (x, z)
