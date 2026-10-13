import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(
        self, num_classes, output_size, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        super().__init__()
        self.num_classes = num_classes
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        # 保持 state_dict 键与参考模型一致
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        # 自定义 one_hot: 使用 new_zeros 和 scatter_ 避免 F.one_hot
        # x 形状 [N, L], int64
        N, L = x.shape
        x_onehot = x.new_zeros((N, L, self.num_classes), dtype=torch.float32)
        x_onehot.scatter_(2, x.unsqueeze(-1).to(torch.int64), 1.0)
        x = x_onehot

        # 参考: x = self.adaptive_avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.permute(0, 2, 1)  # [N, C, L]
        x = cuda_extension.adaptive_avg_pool1d(x, self.output_size)
        x = x.permute(0, 2, 1)  # [N, output_size, C]

        x = x.unsqueeze(1)  # [N, 1, output_size, C]

        # 参考: x = self.conv_transpose(x)
        x = cuda_extension.conv_transpose2d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        return x

    def debug_forward(self, x):
        N, L = x.shape
        x_onehot = x.new_zeros((N, L, self.num_classes), dtype=torch.float32)
        x_onehot.scatter_(2, x.unsqueeze(-1).to(torch.int64), 1.0)
        x = x_onehot
        yield "step_001", x

        x = x.permute(0, 2, 1)
        x = cuda_extension.adaptive_avg_pool1d(x, self.output_size)
        x = x.permute(0, 2, 1)
        yield "step_002", x

        x = x.unsqueeze(1)
        yield "step_003", x

        x = cuda_extension.conv_transpose2d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        yield "step_004", x
        yield "step_005", x
