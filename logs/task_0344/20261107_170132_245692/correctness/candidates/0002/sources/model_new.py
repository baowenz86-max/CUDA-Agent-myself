import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, padding, input_channels, output_channels, kernel_size, bias):
        super().__init__()
        self.padding = padding
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.bias_flag = bias
        # 保持与参考模型相同的 state_dict 键：'conv.0.weight', 'conv.0.bias'
        self.conv_0_weight = nn.Parameter(torch.empty(output_channels, input_channels, kernel_size, kernel_size))
        self.conv_0_bias = nn.Parameter(torch.empty(output_channels))

    def forward(self, x, indices, dim):
        x = cuda_extension.reflection_pad2d_forward(x, self.padding)
        x = cuda_extension.conv2d_cudnn_forward(x, self.conv_0_weight, self.conv_0_bias, stride=1, padding=0)
        x = cuda_extension.gather_custom_forward(x, dim, indices)
        return x

    def debug_forward(self, x, indices, dim):
        x = cuda_extension.reflection_pad2d_forward(x, self.padding)
        yield 'step_001', x
        x = cuda_extension.conv2d_cudnn_forward(x, self.conv_0_weight, self.conv_0_bias, stride=1, padding=0)
        yield 'step_002', x
        x = cuda_extension.gather_custom_forward(x, dim, indices)
        yield 'step_003', x
        yield 'step_004', x
