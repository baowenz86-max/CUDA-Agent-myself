import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # 将输入从 CPU 复制到 CUDA（参考执行设备为 cuda:0）
        x = cuda_extension.copy_to_cuda_forward(x)
        # F.pad(x, (2, 2, 2, 2))
        x = cuda_extension.pad_forward(x, 2)
        # torch.sum(x, dim=self.dim)
        x = cuda_extension.sum_dim_forward(x, self.dim)
        return x

    def debug_forward(self, x):
        x_cuda = cuda_extension.copy_to_cuda_forward(x)
        # step_001: 填充后的张量
        padded = cuda_extension.pad_forward(x_cuda, 2)
        yield "step_001", padded
        # step_002: 求和后的张量
        summed = cuda_extension.sum_dim_forward(padded, self.dim)
        yield "step_002", summed
        # step_003: 最终返回
        yield "step_003", summed
