import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, eps: float = 1e-5):
        super().__init__()
        # 使用 nn.GroupNorm 仅用于参数初始化，确保 state_dict 键一致
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 调用自定义 CUDA 算子完成 GroupNorm 计算
        return cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )

    def debug_forward(self, *inputs):
        """调试接口：按 checkpoints 顺序 yield (id, value)"""
        x = inputs[0]
        output = cuda_extension.group_norm_forward(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.group_norm.num_groups,
            self.group_norm.eps,
        )
        yield 'step_001', output
