import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # 参考模型没有可学习参数或子模块

    def forward(self, x):
        x = cuda_extension.custom_hardtanh(x)
        x = cuda_extension.custom_slice_last_dim(x)
        x = cuda_extension.custom_instance_norm_3d(x)
        return x

    def debug_forward(self, x):
        x1 = cuda_extension.custom_hardtanh(x)
        yield 'step_001', x1
        x2 = cuda_extension.custom_slice_last_dim(x1)
        yield 'step_002', x2
        x3 = cuda_extension.custom_instance_norm_3d(x2)
        yield 'step_003', x3
        yield 'step_004', x3
