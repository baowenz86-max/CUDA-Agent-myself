import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        # 保留原 GroupNorm 模块以维持 state_dict 键，但 forward 中不使用
        self.groupnorm = nn.GroupNorm(num_groups=num_channels, num_channels=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用自定义 CUDA 算子实现 torch.mean(x, dim=1, keepdim=True)
        x = cuda_extension.custom_mean_dim1_keepdim(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        mean_result = cuda_extension.custom_mean_dim1_keepdim(x)
        yield "step_001", mean_result
        yield "step_002", mean_result