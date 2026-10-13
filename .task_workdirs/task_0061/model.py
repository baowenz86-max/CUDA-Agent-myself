import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, module_dict_items):
        super().__init__()
        self.conv_transpose = F.conv_transpose1d
        self.module_dict = nn.ModuleDict(module_dict_items)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x, key):
        x = self.conv_transpose(
            x,
            self.module_dict[key].weight,
            bias=self.module_dict[key].bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        return x


batch_size = 128
in_channels = 3
out_channels = 16
length = 16384  # Increased from 512 to 16384 to increase computation
kernel_size = 3
stride = 2
padding = 1
output_padding = 1

module_dict_items = {
    "conv1": nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
    ),
    "conv2": nn.ConvTranspose1d(
        in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
    ),
}


def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    key = "conv1"  # You can choose 'conv1' or 'conv2' here
    return [x, key]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, module_dict_items]