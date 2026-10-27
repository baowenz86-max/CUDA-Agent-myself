import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding, histc_bins, histc_min, histc_max
    ):
        super().__init__()
        # 保持 state_dict 键与参考模型一致
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.histc_bins = histc_bins
        self.histc_min = histc_min
        self.histc_max = histc_max

    def forward(self, x):
        # 使用自定义 CUDA 算子替代 torch.nn.functional.conv_transpose1d 和 torch.histc
        # 参数与参考调用完全一致
        x = cuda_extension.custom_conv_transpose1d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
            self.conv_transpose.groups,
            self.conv_transpose.dilation[0]
        )
        hist = cuda_extension.custom_histc(
            x,
            self.histc_bins,
            self.histc_min,
            self.histc_max
        )
        return x, hist

    def debug_forward(self, x):
        # 按 checkpoints 顺序 yield (id, value)，计算逻辑与 forward 相同
        x = cuda_extension.custom_conv_transpose1d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0],
            self.conv_transpose.groups,
            self.conv_transpose.dilation[0]
        )
        yield "step_001", x
        hist = cuda_extension.custom_histc(
            x,
            self.histc_bins,
            self.histc_min,
            self.histc_max
        )
        yield "step_002", hist
        yield "step_003", (x, hist)
