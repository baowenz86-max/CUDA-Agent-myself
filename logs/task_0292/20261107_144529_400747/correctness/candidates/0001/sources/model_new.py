import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape, negative_slope):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=negative_slope)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = cuda_extension.conv2d_no_bias(x, self.conv.weight)
        x = cuda_extension.batch_norm2d_inference(x, self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps)
        x = cuda_extension.leaky_relu(x, self.leakyrelu.negative_slope)
        x = cuda_extension.add_bias_broadcast(x, self.bias)
        return x

    def debug_forward(self, x):
        x = cuda_extension.conv2d_no_bias(x, self.conv.weight)
        yield 'step_001', x
        x = cuda_extension.batch_norm2d_inference(x, self.bn.weight, self.bn.bias, self.bn.running_mean, self.bn.running_var, self.bn.eps)
        yield 'step_002', x
        x = cuda_extension.leaky_relu(x, self.leakyrelu.negative_slope)
        yield 'step_003', x
        x = cuda_extension.add_bias_broadcast(x, self.bias)
        yield 'step_004', x
        yield 'step_005', x
