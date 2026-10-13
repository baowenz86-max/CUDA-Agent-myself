import torch
import torch.nn as nn
import torch.nn.functional as F


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
        x = F.one_hot(x, num_classes=self.num_classes).float()
        # 参考: x = self.adaptive_avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        # 对 [N, L, C] -> permute -> [N, C, L] -> adaptive_avg_pool -> [N, C, output_size]
        # 然后 permute 回 [N, output_size, C]
        x = x.permute(0, 2, 1)
        x = torch.ops.cuda_extension.adaptive_avg_pool1d(x, self.output_size)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(1)
        # 参考: x = self.conv_transpose(x)
        x = torch.ops.cuda_extension.conv_transpose2d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        return x

    def debug_forward(self, x):
        x = F.one_hot(x, num_classes=self.num_classes).float()
        yield "step_001", x
        x = x.permute(0, 2, 1)
        x = torch.ops.cuda_extension.adaptive_avg_pool1d(x, self.output_size)
        x = x.permute(0, 2, 1)
        yield "step_002", x
        x = x.unsqueeze(1)
        yield "step_003", x
        x = torch.ops.cuda_extension.conv_transpose2d(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
        )
        yield "step_004", x
        yield "step_005", x
