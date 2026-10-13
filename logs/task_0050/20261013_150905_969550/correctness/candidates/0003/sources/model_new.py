import torch
import torch.nn as nn
import cuda_extension


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
        # 保持与参考模型相同的参数名称和形状，以便 state_dict 兼容
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用自定义 CUDA 扩展算子进行卷积，参数与参考 nn.Conv2d 一致
        return cuda_extension.conv2d_forward(
            x,
            self.conv2d.weight,
            self.conv2d.stride[0],
            self.conv2d.stride[1],
            self.conv2d.padding[0],
            self.conv2d.padding[1],
            self.conv2d.dilation[0],
            self.conv2d.dilation[1],
            self.conv2d.groups,
            self.conv2d.bias,
        )

    def debug_forward(self, x: torch.Tensor):
        # 按 checkpoint 顺序 yield (id, value)
        out = self.forward(x)
        yield "step_001", out
