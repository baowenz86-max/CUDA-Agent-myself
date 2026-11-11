# model_new.py
# 本文件仅调用后续 CUDA/C++ 扩展中会实现的自定义算子。
# 不包含任何 torch 算子实现（无 torch.einsum、F.elu 调用）。
# 与 model.py 的 Model 类保持相同签名、state_dict 键与 forward 接口。

import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # 参考实现没有任何可学习参数，einsum 下标固定为 "ijk->ijk"
        self.einsum = "ijk->ijk"

    def forward(self, x):
        # 步骤 1: torch.einsum(self.einsum, x) 的 identity 语义
        # 自定义算子 custom_identity: 返回与输入相同的张量（支持任意 stride）
        x = cuda_extension.custom_identity(x)
        # 步骤 2: F.elu(x) 默认 alpha=1.0, inplace=False
        # 自定义算子 custom_elu: 逐元素 ELU 计算
        x = cuda_extension.custom_elu(x, alpha=1.0)
        return x

    def debug_forward(self, *inputs):
        """按 task_contract 的 checkpoints 顺序 yield (id, value)，
        包含最终返回。调试接口与正常 forward 使用完全相同的自定义算子，
        不改变计算公式。"""
        x = inputs[0]
        x = cuda_extension.custom_identity(x)
        yield "step_001", x
        x = cuda_extension.custom_elu(x, alpha=1.0)
        yield "step_002", x
        yield "step_003", x
