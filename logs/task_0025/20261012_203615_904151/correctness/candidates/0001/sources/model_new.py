import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 参考实现 self.reflection_pad = nn.ReflectionPad2d(in_channels // 2)
        # 这里保存 pad 数值，实际计算由自定义算子完成
        self.pad = in_channels // 2

    def forward(self, x):
        # 顺序执行三个步骤，与参考一致
        x = cuda_extension.reflection_pad2d_forward(x, self.pad)
        x = cuda_extension.erf_forward(x)
        x = cuda_extension.sin_forward(x)
        return x

    def debug_forward(self, x):
        # 按 checkpoints 顺序 yield (id, value)
        x = cuda_extension.reflection_pad2d_forward(x, self.pad)
        yield 'step_001', x
        x = cuda_extension.erf_forward(x)
        yield 'step_002', x
        x = cuda_extension.sin_forward(x)
        yield 'step_003', x
        yield 'step_004', x
