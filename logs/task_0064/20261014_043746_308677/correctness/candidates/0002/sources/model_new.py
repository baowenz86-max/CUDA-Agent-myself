import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_channels, padding_size, out_channels):
        super().__init__()
        self.padding_size = padding_size
        self.input_channels = input_channels
        self.out_channels = out_channels
        # 保持 state_dict 键与参考模型一致
        self.conv = nn.Conv2d(input_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x = cuda_extension.reflection_pad2d(x, self.padding_size[0], self.padding_size[1], self.padding_size[2], self.padding_size[3])
        x = cuda_extension.conv2d(x, self.conv.weight, self.conv.bias, 1, 1, 0, 0, 1, 1)
        x = cuda_extension.nansum_dim12(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.reflection_pad2d(x, self.padding_size[0], self.padding_size[1], self.padding_size[2], self.padding_size[3])
        yield 'step_001', x
        x = cuda_extension.conv2d(x, self.conv.weight, self.conv.bias, 1, 1, 0, 0, 1, 1)
        yield 'step_002', x
        x = cuda_extension.nansum_dim12(x)
        yield 'step_003', x
        yield 'step_004', x
