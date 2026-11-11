import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        # 确保输入在 GPU 上；绑定层会检查设备。
        x1 = x1.cuda()
        x2 = x2.cuda()
        x = cuda_extension.custom_cat(x1, x2, self.dim)
        equal = cuda_extension.custom_equal(x1, x2)
        return x, equal

    def debug_forward(self, *inputs):
        # 与 forward 使用相同的计算逻辑，但按 checkpoint 顺序 yield (id, value)
        x1, x2 = inputs
        x1 = x1.cuda()
        x2 = x2.cuda()
        x = cuda_extension.custom_cat(x1, x2, self.dim)
        yield 'step_001', x
        equal = cuda_extension.custom_equal(x1, x2)
        yield 'step_002', equal
        yield 'step_003', (x, equal)
