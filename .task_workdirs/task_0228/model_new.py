import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, eps_value):
        super().__init__()
        # 保持与参考相同的模块结构，但 forward 中不使用 torch 模块计算
        self.batch_norm = nn.BatchNorm2d(num_channels, eps=eps_value)

    def forward(self, x):
        # 第一步：上三角掩码（自定义算子）
        x = cuda_extension.triu_forward(x)
        # 第二步：BatchNorm2d eval 模式（自定义算子）
        x = cuda_extension.batch_norm_eval_forward(
            x,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.eps,
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        # step_001: torch.triu
        x_triu = cuda_extension.triu_forward(x)
        yield 'step_001', x_triu
        # step_002: batch_norm eval
        x_bn = cuda_extension.batch_norm_eval_forward(
            x_triu,
            self.batch_norm.running_mean,
            self.batch_norm.running_var,
            self.batch_norm.weight,
            self.batch_norm.bias,
            self.batch_norm.eps,
        )
        yield 'step_002', x_bn
        # step_003: return
        yield 'step_003', x_bn
