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
        conv_out = cuda_extension.conv_transpose1d_no_bias(
            x,
            self.conv_transpose.weight,
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
            output_padding=self.conv_transpose.output_padding[0]
        )
        out = conv_out
        for param in self.params:
            out = cuda_extension.add_channel_bias_round(out, param)
        return out

    def debug_forward(self, x):
        conv_out = cuda_extension.conv_transpose1d_no_bias(
            x,
            self.conv_transpose.weight,
            stride=self.conv_transpose.stride[0],
            padding=self.conv_transpose.padding[0],
            output_padding=self.conv_transpose.output_padding[0]
        )
        yield "step_001", conv_out
        out = conv_out
        for i, param in enumerate(self.params):
            out = cuda_extension.add_channel_bias_round(out, param)
            yield f"step_{i+2:03d}", out
        return out
