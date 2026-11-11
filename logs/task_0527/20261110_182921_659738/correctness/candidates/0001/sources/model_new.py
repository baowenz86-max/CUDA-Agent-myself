import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, input_size, kernel_size, dilation, padding):
        super().__init__()
        # 保持与参考相同的参数，不创建任何 state_dict 条目（参考模型 state_dict 为空）
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding

    def forward(self, x):
        # 调用自定义扩展算子，避免在 Python 中使用 torch 计算
        max_val = torch.ops.cuda_extension.unfold_amax_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.dilation[0],
            self.dilation[1],
            self.padding[0],
            self.padding[1]
        )
        non_zero_count = torch.ops.cuda_extension.unfold_count_nonzero_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.dilation[0],
            self.dilation[1],
            self.padding[0],
            self.padding[1]
        )
        return max_val, non_zero_count

    def debug_forward(self, *inputs):
        # 按 checkpoints 顺序 yield (id, value)，包括最终返回
        # 注意：debug_forward 不能改变正常 forward 的计算公式
        x = inputs[0]
        # step_001: unfold 中间结果（仅调试时显式构造）
        unfolded = torch.ops.cuda_extension.unfold_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.dilation[0],
            self.dilation[1],
            self.padding[0],
            self.padding[1]
        )
        yield 'step_001', unfolded
        # step_002: amax 结果
        max_val = torch.ops.cuda_extension.unfold_amax_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.dilation[0],
            self.dilation[1],
            self.padding[0],
            self.padding[1]
        )
        yield 'step_002', max_val
        # step_003: count_nonzero 结果
        non_zero_count = torch.ops.cuda_extension.unfold_count_nonzero_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.dilation[0],
            self.dilation[1],
            self.padding[0],
            self.padding[1]
        )
        yield 'step_003', non_zero_count
        # step_004: 最终返回 tuple
        yield 'step_004', (max_val, non_zero_count)
