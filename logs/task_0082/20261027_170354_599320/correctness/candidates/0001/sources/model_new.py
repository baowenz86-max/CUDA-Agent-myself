import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, upscale_factor):
        super().__init__()
        self.deconv_weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size, kernel_size))
        self.deconv_bias = nn.Parameter(torch.zeros(out_channels))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.upscale_factor = upscale_factor

    def forward(self, x):
        x = cuda_extension.deconv_forward(x, self.deconv_weight, self.deconv_bias, self.kernel_size, self.stride, self.padding, self.output_padding)
        x = cuda_extension.pixel_shuffle_forward(x, self.upscale_factor)
        x = cuda_extension.relu_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.deconv_forward(x, self.deconv_weight, self.deconv_bias, self.kernel_size, self.stride, self.padding, self.output_padding)
        yield 'step_001', x
        x = cuda_extension.pixel_shuffle_forward(x, self.upscale_factor)
        yield 'step_002', x
        x = cuda_extension.relu_forward(x)
        yield 'step_003', x
        yield 'step_004', x
