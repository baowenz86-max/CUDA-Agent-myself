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
        bias: bool = True,
    ):
        super().__init__()
        # 保持与参考模型相同的结构，只使用传入的 kernel_size, stride, padding, dilation
        self.maxpool = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用自定义算子实现 maxpool 和 relu
        # 从 self.maxpool 提取参数
        kernel_h, kernel_w = self.maxpool.kernel_size
        stride = self.maxpool.stride
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        padding = self.maxpool.padding
        padding_h, padding_w = padding if isinstance(padding, tuple) else (padding, padding)
        dilation = self.maxpool.dilation
        dilation_h, dilation_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        ceil_mode = self.maxpool.ceil_mode

        x = cuda_extension.custom_maxpool2d(
            x,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            ceil_mode,
        )
        x = cuda_extension.custom_relu(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        kernel_h, kernel_w = self.maxpool.kernel_size
        stride = self.maxpool.stride
        stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
        padding = self.maxpool.padding
        padding_h, padding_w = padding if isinstance(padding, tuple) else (padding, padding)
        dilation = self.maxpool.dilation
        dilation_h, dilation_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        ceil_mode = self.maxpool.ceil_mode

        x = cuda_extension.custom_maxpool2d(
            x,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            ceil_mode,
        )
        yield "step_001", x

        x = cuda_extension.custom_relu(x)
        yield "step_002", x

        yield "step_003", x
