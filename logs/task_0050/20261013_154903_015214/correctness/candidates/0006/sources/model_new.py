import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = False,
    ):
        super().__init__()
        # 保持与参考模型完全一致的 state_dict 键：conv2d.weight
        self.conv2d = nn.Module()
        self.conv2d.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // 1, *kernel_size)
        )
        # 初始化方式与 nn.Conv2d 默认相同，保证验证通过
        nn.init.kaiming_uniform_(self.conv2d.weight, a=5 ** 0.5)
        # 存储卷积参数供 forward 使用，不调用 torch 卷积
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
        self.groups = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.conv2d_forward(
            self.conv2d.weight,
            x,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
        )

    def debug_forward(self, x: torch.Tensor):
        out = self.forward(x)
        yield 'step_001', out
