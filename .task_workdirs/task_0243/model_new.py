import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, avg_pool_kernel, conv_padding):
        super().__init__()
        self.avg_pool_kernel = avg_pool_kernel
        self.conv_padding = conv_padding
        self.stride = stride
        # 创建 Conv3d 参数以保持 state_dict 键
        self.conv3d_weight = nn.Parameter(torch.empty(out_channels, in_channels, *kernel_size))
        self.conv3d_bias = nn.Parameter(torch.empty(out_channels))
        # 按参考实现初始化参数（与 nn.Conv3d 默认初始化一致）
        nn.init.kaiming_uniform_(self.conv3d_weight, a=5**0.5)
        fan_in = in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2]
        bound = 1 / fan_in**0.5 if fan_in > 0 else 0
        nn.init.uniform_(self.conv3d_bias, -bound, bound)

    def forward(self, x):
        x = cuda_extension.avg_pool3d_forward(x, self.avg_pool_kernel)
        return cuda_extension.conv3d_forward(x, self.conv3d_weight, self.conv3d_bias, self.stride, self.conv_padding)

    def debug_forward(self, x):
        x = cuda_extension.avg_pool3d_forward(x, self.avg_pool_kernel)
        yield 'step_001', x
        x = cuda_extension.conv3d_forward(x, self.conv3d_weight, self.conv3d_bias, self.stride, self.conv_padding)
        yield 'step_002', x
