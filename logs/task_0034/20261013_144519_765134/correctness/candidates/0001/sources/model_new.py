import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self, num_classes, in_channels, out_channels, kernel_size, stride, padding, output_padding, scale_factor
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.num_classes = num_classes
        # 存储标量参数供自定义算子使用
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scale_factor = scale_factor

    def forward(self, x, target):
        # 确保输入在 CUDA 设备上（与参考模型在 CUDA 上运行一致）
        if not x.is_cuda:
            x = x.cuda()
        if not target.is_cuda:
            target = target.cuda()

        # 调用自定义 conv_transpose 算子（cuDNN）
        x = cuda_extension.custom_conv_transpose2d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )

        # 调用自定义 nearest upsample 算子
        x = cuda_extension.custom_upsample_nearest2d(
            x,
            self.scale_factor,
        )

        # 调用自定义 one_hot 算子
        target = cuda_extension.custom_one_hot(
            target,
            self.num_classes,
        )

        return x, target

    def debug_forward(self, x, target):
        # 确保输入在 CUDA 设备上
        if not x.is_cuda:
            x = x.cuda()
        if not target.is_cuda:
            target = target.cuda()

        # step_001: conv_transpose
        x = cuda_extension.custom_conv_transpose2d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        yield "step_001", x

        # step_002: upsample
        x = cuda_extension.custom_upsample_nearest2d(
            x,
            self.scale_factor,
        )
        yield "step_002", x

        # step_003: one_hot
        target = cuda_extension.custom_one_hot(
            target,
            self.num_classes,
        )
        yield "step_003", target

        # step_004: final return (x, target)
        yield "step_004", (x, target)
