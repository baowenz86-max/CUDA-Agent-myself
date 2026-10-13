import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        num_features,
        out_features,
    ):
        super().__init__()
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.linear = nn.Linear(num_features, out_features)

    def forward(self, x):
        x = self.conv_transpose1d(x)
        x = x + self.bias
        x = torch.roll(x, shifts=1, dims=2)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the linear layer
        x = F.linear(x, self.linear.weight, self.linear.bias)
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
in_channels = 8
out_channels = 16
length = 2048  # Increased to increase computation time
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1)
num_features = out_channels * (length * stride)  # Calculated based on the output size of ConvTranspose1d
out_features = 10


def get_inputs():
    return [torch.randn(batch_size, in_channels, length)]


def get_init_inputs():
    return [
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
        num_features,
        out_features,
    ]