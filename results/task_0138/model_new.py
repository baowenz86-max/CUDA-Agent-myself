import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, size, p, dim):
        super(ModelNew, self).__init__()
        # 保持与原模型相同，但此处不需要任何参数，因为forward不使用self中的张量。
        # 原模型在__init__中创建了diag和norm但未使用，state_dict为空，所以这里也保持为空。
        # 注意：原模型创建了self.diag和self.norm，但未注册为Parameter，所以state_dict为空。
        # 为了完全保持state_dict一致性，我们也可以创建相同的普通属性，但不影响。
        # 这里不创建任何参数。

    def forward(self, x):
        x = cuda_extension.diag_embed(x)
        x = cuda_extension.l1_norm_dim1(x)
        return x

    def debug_forward(self, x):
        # 逐步计算，产生检查点
        x1 = cuda_extension.diag_embed(x)
        yield 'step_001', x1
        x2 = cuda_extension.l1_norm_dim1(x1)
        yield 'step_002', x2
        yield 'step_003', x2
