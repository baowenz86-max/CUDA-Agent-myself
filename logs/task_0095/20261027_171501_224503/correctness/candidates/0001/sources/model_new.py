import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, start, end, steps):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.start = start
        self.end = end
        self.steps = steps

    def forward(self, x):
        linear_space = cuda_extension.create_linspace(self.start, self.end, self.steps, x.device)
        x = cuda_extension.tensor_sub(x, linear_space)
        x = cuda_extension.conv_transpose1d(x, self.conv_transpose.weight, self.conv_transpose.bias, self.conv_transpose.stride, self.conv_transpose.padding, self.conv_transpose.output_padding)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        linear_space = cuda_extension.create_linspace(self.start, self.end, self.steps, x.device)
        yield 'step_001', linear_space
        x = cuda_extension.tensor_sub(x, linear_space)
        yield 'step_002', x
        x = cuda_extension.conv_transpose1d(x, self.conv_transpose.weight, self.conv_transpose.bias, self.conv_transpose.stride, self.conv_transpose.padding, self.conv_transpose.output_padding)
        yield 'step_003', x
        yield 'step_004', x
