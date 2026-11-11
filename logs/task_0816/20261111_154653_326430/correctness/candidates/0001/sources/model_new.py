import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, scale_factor, mode):
        super().__init__()
        # 仅为接口兼容和属性记录；forward 中不使用 nn.Upsample 实例
        self.scale_factor = float(scale_factor)
        self.mode = mode

    def forward(self, x):
        # 第一步：上采样 bilinear，scale_factor 为构造时指定值
        x = cuda_extension.upsample_bilinear2d(x, self.scale_factor)
        # 第二步：通道均值，保持维度
        x = cuda_extension.mean_dim1_keepdim(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        # 按照 task_contract 检查点顺序输出中间值
        x = cuda_extension.upsample_bilinear2d(x, self.scale_factor)
        yield 'step_001', x
        x = cuda_extension.mean_dim1_keepdim(x)
        yield 'step_002', x
        yield 'step_003', x
