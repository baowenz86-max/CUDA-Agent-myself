import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim, input_shape1, input_shape2):
        super().__init__()
        self.dim = dim

    def forward(self, x1, x2):
        return cuda_extension.cat_prod_forward(x1, x2, self.dim)

    def debug_forward(self, x1, x2):
        # 参考步骤：x = torch.cat((x1, x2), dim=self.dim)
        # 使用自定义算子的中间返回，这里假设 cat_prod_forward 内部可返回拼接结果
        # 目前设计为仅返回乘积结果，为满足 debug 检查点，需要显式获取拼接中间结果。
        # 因此定义辅助算子 cat_forward 返回拼接张量，再由 cat_prod_forward 使用该拼接结果。
        cat = cuda_extension.cat_forward(x1, x2, self.dim)
        yield 'step_001', cat
        prod = cuda_extension.prod_forward(cat, self.dim)
        yield 'step_002', prod
        yield 'step_003', prod
