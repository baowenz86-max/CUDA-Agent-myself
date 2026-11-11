import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, padding_size, num_classes):
        super().__init__()
        self.zero_pad = nn.ZeroPad2d(padding_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.num_classes = num_classes

    def forward(self, x):
        x = cuda_extension.zero_pad2d_forward(x, padding=1)
        x = cuda_extension.log_softmax_channel_forward(x, dim=1)
        x = cuda_extension.conv_transpose2d_forward(x, self.conv_transpose.weight, self.conv_transpose.bias, stride=2, padding=1, output_padding=0)
        return x

    def debug_forward(self, x):
        x = cuda_extension.zero_pad2d_forward(x, padding=1)
        yield "step_001", x
        x = cuda_extension.log_softmax_channel_forward(x, dim=1)
        yield "step_002", x
        x = cuda_extension.conv_transpose2d_forward(x, self.conv_transpose.weight, self.conv_transpose.bias, stride=2, padding=1, output_padding=0)
        yield "step_003", x
        yield "step_004", x
