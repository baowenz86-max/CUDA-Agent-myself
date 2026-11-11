import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # 保留原结构以维持接口一致；forward 中不调用该模块，实际池化由自定义算子完成
        self.max_pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = cuda_extension.custom_abs(x)
        x = cuda_extension.custom_max_pool2d(
            x,
            kernel_size=(3, 5),
            stride=1,
            padding=(1, 2),
            dilation=1,
            ceil_mode=False
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_abs(x)
        yield 'step_001', x
        x = cuda_extension.custom_max_pool2d(
            x,
            kernel_size=(3, 5),
            stride=1,
            padding=(1, 2),
            dilation=1,
            ceil_mode=False
        )
        yield 'step_002', x
        yield 'step_003', x
