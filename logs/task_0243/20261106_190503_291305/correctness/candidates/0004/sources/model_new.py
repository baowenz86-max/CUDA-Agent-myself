import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, avg_pool_kernel, conv_padding):
        super().__init__()
        self.avg_pool_kernel = avg_pool_kernel
        self.conv_padding = conv_padding
        self.stride = stride
        # 创建与参考 nn.Conv3d 相同的参数字典，保持 state_dict 键
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=conv_padding)

    def forward(self, x):
        x = cuda_extension.avg_pool3d_forward(x, self.avg_pool_kernel[0])
        return cuda_extension.conv3d_forward(x, self.conv3d.weight, self.conv3d.bias, self.stride, self.conv_padding)

    def debug_forward(self, x):
        x = cuda_extension.avg_pool3d_forward(x, self.avg_pool_kernel[0])
        yield 'step_001', x
        x = cuda_extension.conv3d_forward(x, self.conv3d.weight, self.conv3d.bias, self.stride, self.conv_padding)
        yield 'step_002', x
