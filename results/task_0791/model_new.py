import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def forward(self, x1, x2):
        return cuda_extension.logical_or_round(x1, x2)

    def debug_forward(self, *inputs):
        x1, x2 = inputs
        # 调用自定义算子得到最终 float32 结果
        out = cuda_extension.logical_or_round(x1, x2)
        # 为了提供 step_001 的 bool 中间值，计算逻辑或（仅用于调试，不改变 forward 路径）
        # 这里复用 cuda_extension.logical_or_round 的结果，但需要 bool 类型，因此通过比较得到 bool。
        # 由于 out 是 float32 且值为 0.0 或 1.0，大于 0.5 即为 True
        bool_out = out > 0.5
        yield 'step_001', bool_out
        yield 'step_002', out
        yield 'step_003', out
