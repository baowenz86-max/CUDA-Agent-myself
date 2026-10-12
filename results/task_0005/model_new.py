import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        # Store parameters exactly as original to preserve state_dict keys.
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        # Use custom CUDA ops corresponding to each reference operation.
        x = cuda_extension.log10_forward(x)                       # torch.log10
        x = cuda_extension.lt_scalar_forward(x, 0.5)              # torch.lt(x, 0.5) -> bool tensor
        x = cuda_extension.boolean_to_float(x)                    # x.float() conversion
        x = cuda_extension.conv_transpose1d_forward(
            x, self.conv_transpose1d.weight, self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0], self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )                                                         # ConvTranspose1d forward
        x = cuda_extension.add_bias_forward(x, self.bias)         # + self.bias
        return x

    def debug_forward(self, x):
        # Yield (id, value) for each checkpoint, using the same operations.
        x = cuda_extension.log10_forward(x)
        yield 'step_001', x
        x = cuda_extension.lt_scalar_forward(x, 0.5)
        yield 'step_002', x
        x = cuda_extension.boolean_to_float(x)
        x = cuda_extension.conv_transpose1d_forward(
            x, self.conv_transpose1d.weight, self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0], self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        yield 'step_003', x
        x = cuda_extension.add_bias_forward(x, self.bias)
        yield 'step_004', x
        yield 'step_005', x
