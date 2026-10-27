import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_dim, kernel_size, stride, padding, output_padding, bias):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            input_dim, input_dim, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )

    def forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x, self.conv_transpose1d.weight,
            self.conv_transpose1d.kernel_size[0],
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        x = cuda_extension.ge_forward(x, 0.0)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv_transpose1d_forward(
            x, self.conv_transpose1d.weight,
            self.conv_transpose1d.kernel_size[0],
            self.conv_transpose1d.stride[0],
            self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        yield 'step_001', x
        x = cuda_extension.ge_forward(x, 0.0)
        yield 'step_002', x
        yield 'step_003', x
