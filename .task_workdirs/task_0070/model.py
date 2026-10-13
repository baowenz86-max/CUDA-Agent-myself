import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        # Define the convolutional layer with given parameters
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)  # Apply ReLU activation function
        x = self.pool(x)  # Apply max pooling
        return x


# Define the input data shape and parameters for the layer
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 512, 512  # Increased width to 512 to increase computation
kernel_size = 3
stride = 2
padding = 1


def get_inputs():
    # Create random tensor as input
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    # Return the parameters needed for Model initialization
    return [in_channels, out_channels, kernel_size, stride, padding]