import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, upscale_factor):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.upscale_factor = upscale_factor

    def forward(self, x):
        x = cuda_extension.deconv_forward(x, self.deconv.weight, self.deconv.bias, self.kernel_size, self.stride, self.padding, self.output_padding)
        x = cuda_extension.pixel_shuffle_forward(x, self.upscale_factor)
        x = cuda_extension.relu_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.deconv_forward(x, self.deconv.weight, self.deconv.bias, self.kernel_size, self.stride, self.padding, self.output_padding)
        yield 'step_001', x
        x = cuda_extension.pixel_shuffle_forward(x, self.upscale_factor)
        yield 'step_002', x
        x = cuda_extension.relu_forward(x)
        yield 'step_003', x
        yield 'step_004', x
