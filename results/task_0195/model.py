import torch


class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, input_dim):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        self.input_dim = input_dim

    def forward(self, x):
        assert x.dim() == self.input_dim, f"Expected input dim to be {self.input_dim}, but got {x.dim()}"
        x = torch.nn.functional.relu(
            torch.nn.functional.conv2d(
                x,
                self.conv2d.weight,
                bias=self.conv2d.bias,
                stride=self.conv2d.stride,
                padding=self.conv2d.padding,
                dilation=self.conv2d.dilation,
                groups=self.conv2d.groups,
            )
        )
        return x


batch_size = 512
in_channels = 3
out_channels = 16
height, width = 128, 128  # Increased from 32x32 to 128x128
kernel_size = 3
stride = 1
padding = 1
dilation = 1
bias = True
input_dim = 4


def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation, bias, input_dim]