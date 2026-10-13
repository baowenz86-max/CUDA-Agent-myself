import math
import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self, num_classes, in_channels, out_channels, kernel_size, stride, padding, output_padding, scale_factor
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scale_factor = scale_factor

        # 保持与 nn.ConvTranspose2d 相同的参数初始化方式，确保 state_dict 键一致
        self.conv_transpose_weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.conv_transpose_bias = nn.Parameter(torch.empty(out_channels))

        # 使用与参考模型相同的初始化
        nn.init.kaiming_uniform_(self.conv_transpose_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_transpose_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv_transpose_bias, -bound, bound)

    def forward(self, x, target):
        # 使用自定义 CUDA 扩展算子实现 ConvTranspose2d
        x = cuda_extension.conv_transpose_forward(
            x,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        # 使用自定义 CUDA 扩展算子实现 nearest upsample
        x = cuda_extension.upsample_nearest_forward(x, self.scale_factor)
        # 使用自定义 CUDA 扩展算子实现 one_hot
        target = cuda_extension.one_hot_forward(target, self.num_classes)
        return x, target

    def debug_forward(self, x, target):
        x = cuda_extension.conv_transpose_forward(
            x,
            self.conv_transpose_weight,
            self.conv_transpose_bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        yield "step_001", x
        x = cuda_extension.upsample_nearest_forward(x, self.scale_factor)
        yield "step_002", x
        target = cuda_extension.one_hot_forward(target, self.num_classes)
        yield "step_003", target
        yield "step_004", (x, target)
