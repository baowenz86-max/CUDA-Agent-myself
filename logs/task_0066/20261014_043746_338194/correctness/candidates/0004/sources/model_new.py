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
        self.relu6 = nn.ReLU6()

    def forward(self, x, y):
        x = cuda_extension.conv_transpose1d_forward(
            x, self.conv_transpose1d.weight, self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0], self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        mask = cuda_extension.ge_forward(x, self.threshold)
        x = cuda_extension.relu6_forward(x)
        x = cuda_extension.mul_forward(x, mask.float())
        return x

    def debug_forward(self, x, y):
        x = cuda_extension.conv_transpose1d_forward(
            x, self.conv_transpose1d.weight, self.conv_transpose1d.bias,
            self.conv_transpose1d.stride[0], self.conv_transpose1d.padding[0],
            self.conv_transpose1d.output_padding[0]
        )
        yield 'step_001', x
        mask = cuda_extension.ge_forward(x, self.threshold)
        yield 'step_002', mask
        x = cuda_extension.relu6_forward(x)
        yield 'step_003', x
        x = cuda_extension.mul_forward(x, mask.float())
        yield 'step_004', x
        yield 'step_005', x
