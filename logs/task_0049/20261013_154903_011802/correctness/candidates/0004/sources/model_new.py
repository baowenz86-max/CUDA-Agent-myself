import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, num_params):
        super().__init__()
        # Mirror ConvTranspose1d's weight shape: (in_channels, out_channels, kernel_size)
        self.conv_transpose = nn.Module()
        self.conv_transpose.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size))
        self.conv_transpose.bias = None  # Reference never uses conv bias
        self.conv_transpose.stride = (stride,) if isinstance(stride, int) else stride
        self.conv_transpose.padding = (padding,) if isinstance(padding, int) else padding
        self.conv_transpose.output_padding = (output_padding,) if isinstance(output_padding, int) else output_padding
        self.params = nn.ParameterList([nn.Parameter(torch.randn(out_channels)) for _ in range(num_params)])

    def forward(self, x):
        x = cuda_extension.conv_transpose1d_no_bias(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.stride,
            self.conv_transpose.padding,
            self.conv_transpose.output_padding,
        )
        for param in self.params:
            x = cuda_extension.add_channel_bias(x, param)
        x = cuda_extension.round(x)
        return x
