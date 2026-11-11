import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        # 保留与参考模型相同的参数结构，确保 state_dict 键一致
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = torch.ops.cuda_extension.softmax_channel_last_placeholder(x, 1)
        x = torch.ops.cuda_extension.leaky_relu_placeholder(x, 0.01)
        x = torch.ops.cuda_extension.conv2d_cudnn_placeholder(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding)
        x = torch.ops.cuda_extension.add_bias_placeholder(x, self.bias)
        return x

    def debug_forward(self, x):
        x = torch.ops.cuda_extension.softmax_channel_last_placeholder(x, 1)
        yield "step_001", x
        x = torch.ops.cuda_extension.leaky_relu_placeholder(x, 0.01)
        yield "step_002", x
        x = torch.ops.cuda_extension.conv2d_cudnn_placeholder(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding)
        yield "step_003", x
        x = torch.ops.cuda_extension.add_bias_placeholder(x, self.bias)
        yield "step_004", x
        yield "step_005", x
