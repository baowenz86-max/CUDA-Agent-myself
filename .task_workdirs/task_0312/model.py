import torch
import torch.nn as nn


# Define the model
class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int):
        super(Model, self).__init__()
        # Define a convolutional layer using functional conv1d
        self.conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        # Define a max pooling layer for 1D data
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolutional layer
        x = self.conv1d(x)
        # Apply max pooling layer
        x = self.maxpool1d(x)
        return x


# Define a function to get the input data for the model
def get_inputs():
    batch_size, in_channels, width = 128, 3, 16384  # Increased width from 512 to 16384
    x = torch.randn(batch_size, in_channels, width)
    return [x]


# Define a function to get the initialization parameters for the model
def get_init_inputs():
    in_channels, out_channels, kernel_size, stride, padding, dilation = 3, 64, 3, 1, 1, 1
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]