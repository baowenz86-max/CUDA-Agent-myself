import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_channels, num_classes, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, num_classes, kernel_size, stride=stride, padding=padding, bias=bias)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y):
        x = self.conv(x)
        x = torch.bitwise_xor(x.type(torch.uint8), y.type(torch.uint8)).type(torch.float32)
        x = self.log_softmax(x)
        return x


batch_size = 512
input_channels = 3
num_classes = 10
height, width = 128, 128  # Increased from 32x32 to 128x128 to increase runtime
kernel_size = 3
stride = 1
padding = 1
bias = True


def get_inputs():
    x = torch.randn(batch_size, input_channels, height, width)
    y = torch.randint(0, 2, (batch_size, num_classes, height, width))
    return [x, y]


def get_init_inputs():
    return [input_channels, num_classes, kernel_size, stride, padding, bias]