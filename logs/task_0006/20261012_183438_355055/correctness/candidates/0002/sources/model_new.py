import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, padding_size):
        super().__init__()
        # 原始模型仅使用 ReflectionPad2d，无参数；保留 padding_size 供 forward 使用
        self.padding_size = padding_size

    def forward(self, x):
        # 步骤1: reflection padding
        x = cuda_extension.reflection_pad2d_forward(x, self.padding_size)
        # 步骤2: 大于等于0比较
        x = cuda_extension.ge_forward(x, 0)
        # 步骤3: 取余
        x = cuda_extension.remainder_forward(x, 2)
        return x

    def debug_forward(self, x):
        # 生成器，按参考赋值顺序 yield 各个中间结果和最终结果
        x = cuda_extension.reflection_pad2d_forward(x, self.padding_size)
        yield 'step_001', x
        x = cuda_extension.ge_forward(x, 0)
        yield 'step_002', x
        x = cuda_extension.remainder_forward(x, 2)
        yield 'step_003', x
        yield 'step_004', x
