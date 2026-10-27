import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # 保留 conv 模块以维持 state_dict 键与参考一致
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        x = cuda_extension.trunc_forward(x)
        x = cuda_extension.max_pool2d_forward(x, kernel_size=2, stride=2)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.trunc_forward(x)
        yield 'step_001', x
        x = cuda_extension.max_pool2d_forward(x, kernel_size=2, stride=2)
        yield 'step_002', x
        yield 'step_003', x