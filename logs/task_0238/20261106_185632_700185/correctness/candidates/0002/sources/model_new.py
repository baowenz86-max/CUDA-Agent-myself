import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        # 显式保留参数，确保 state_dict 键与参考一致
        self.weight = self.conv_transpose.weight
        self.bias = self.conv_transpose.bias

    def forward(self, x, target):
        x = cuda_extension.conv_transpose1d_forward(
            x, self.weight, self.bias,
            self.conv_transpose.in_channels,
            self.conv_transpose.out_channels,
            self.conv_transpose.kernel_size[0],
            self.conv_transpose.stride[0],
            self.conv_transpose.padding[0],
            self.conv_transpose.output_padding[0]
        )
        x = cuda_extension.clamp_log2_forward(x, 1e-7)
        loss = cuda_extension.mse_loss_forward(x, target)
        return loss
