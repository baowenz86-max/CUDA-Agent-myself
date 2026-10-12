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
            self.conv_transpose1d.stride,
            self.conv_transpose1d.padding,
            self.conv_transpose1d.output_padding,
        )
        median_value = cuda_extension.median_forward(x)
        median_scalar = float(median_value.detach().cpu())
        x = cuda_extension.ne_float_scalar_forward(x, median_scalar)
        z = cuda_extension.ne_float_scalar_forward(z, float(self.threshold))
        return x, z

    def debug_forward(self, x, z):
        x = self._forward(x, z)
        yield 'step_001', x
        median_value = self._median_value
        yield 'step_002', median_value
        x = self._out_x
        yield 'step_003', x
        z = self._out_z
        yield 'step_004', z
        yield 'step_005', (x, z)
        return
