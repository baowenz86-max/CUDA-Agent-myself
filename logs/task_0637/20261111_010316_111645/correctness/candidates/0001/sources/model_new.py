import torch
import torch.nn as nn
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        # 参考模型没有任何参数，因此这里也不保留额外 state_dict 项。

    def forward(self, x):
        # 第一次 max：沿 dim=1，保留维度
        max_val = cuda_extension.max_forward(x, 1, True)
        # 第二次 max：沿 dim=2，不保留维度
        max_val = cuda_extension.max_forward(max_val.values, 2, False)
        # max_val 是 torch.return_types.max，最终返回其中的 values 张量
        return max_val.values

    def debug_forward(self, x):
        # 与 forward 使用完全相同的计算公式，只是额外暴露中间检查点。
        step_001 = cuda_extension.max_forward(x, 1, True)
        yield 'step_001', step_001
        step_002 = cuda_extension.max_forward(step_001.values, 2, False)
        yield 'step_002', step_002
        yield 'step_003', step_002.values
