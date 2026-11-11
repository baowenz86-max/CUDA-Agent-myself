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
        x = self.conv_transpose(x)
        x = torch.log2(torch.clamp(x, min=1e-7))
        loss = self.mse_loss(x, target)
        return loss
