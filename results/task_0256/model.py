import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, padding_size, num_classes):
        super().__init__()
        self.zero_pad = nn.ZeroPad2d(padding_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.zero_pad(x)
        x = self.log_softmax(x)
        x = self.conv_transpose(x)
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 34, 34  # Increased from 32x32 to 34x34 to slightly increase computation
padding_size = (1, 1, 1, 1)
num_classes = 10


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, padding_size, num_classes]