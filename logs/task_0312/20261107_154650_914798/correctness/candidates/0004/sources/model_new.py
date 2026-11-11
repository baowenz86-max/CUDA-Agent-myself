import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.conv1d_cudnn_forward(
            x, self.conv1d.weight, self.conv1d.bias,
            stride=self.conv1d.stride[0], padding=self.conv1d.padding[0],
            dilation=self.conv1d.dilation[0]
        )
        x = cuda_extension.maxpool1d_forward(
            x, kernel_size=2, stride=2, padding=0, dilation=1
        )
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        conv_out = cuda_extension.conv1d_cudnn_forward(
            x, self.conv1d.weight, self.conv1d.bias,
            stride=self.conv1d.stride[0], padding=self.conv1d.padding[0],
            dilation=self.conv1d.dilation[0]
        )
        yield ("step_001", conv_out)

        maxpool_out = cuda_extension.maxpool1d_forward(
            conv_out, kernel_size=2, stride=2, padding=0, dilation=1
        )
        yield ("step_002", maxpool_out)

        yield ("step_003", maxpool_out)
