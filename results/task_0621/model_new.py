import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # 保留与参考完全相同的模块结构和参数，确保 state_dict 键一致（无参数）
        self.abs = nn.Sequential(nn.AvgPool2d(kernel_size, stride=stride, padding=padding))

    def forward(self, x):
        x = cuda_extension.custom_abs(x)
        x = cuda_extension.custom_avg_pool2d(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_abs(x)
        yield 'step_001', x
        x = cuda_extension.custom_avg_pool2d(x)
        yield 'step_002', x
        yield 'step_003', x
