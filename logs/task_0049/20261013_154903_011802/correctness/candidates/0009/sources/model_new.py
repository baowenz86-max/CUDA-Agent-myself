import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, num_params):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.params = nn.ParameterList([nn.Parameter(torch.randn(out_channels)) for _ in range(num_params)])

    def forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
        )
        for param in self.params:
            x = cuda_extension.bias_add_forward(x, param)
        x = cuda_extension.round_elementwise_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
        )
        yield 'step_001', x
        for i, param in enumerate(self.params):
            x = cuda_extension.bias_add_forward(x, param)
            yield f'step_{i+2:03d}', x
        x = cuda_extension.round_elementwise_forward(x)
        yield 'step_005', x
