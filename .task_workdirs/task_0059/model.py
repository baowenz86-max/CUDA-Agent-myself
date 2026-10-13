import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self, num_classes, output_size, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        super().__init__()
        self.num_classes = num_classes
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )

    def forward(self, x):
        x = F.one_hot(x, num_classes=self.num_classes).float()
        x = self.adaptive_avg_pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.conv_transpose(x)
        return x


batch_size = 256
num_classes = 100
output_size = 1024
in_channels = 1
out_channels = 3
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
sequence_length = 2048  # Increased to make input processing more computationally heavy


def get_inputs():
    return [torch.randint(0, num_classes, (batch_size, sequence_length))]


def get_init_inputs():
    return [num_classes, output_size, in_channels, out_channels, kernel_size, stride, padding, output_padding]