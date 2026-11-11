import torch
import torch.nn as nn


# Define the model
class Model(nn.Module):
    def __init__(self, in_features, out_features, max_pool_kernel_size):
        super().__init__()
        self.hardtanh = nn.Hardtanh()
        self.max_pool = nn.MaxPool2d(kernel_size=max_pool_kernel_size)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.hardtanh(x)
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


# Hyperparameters
batch_size = 512
in_channels = 1
height, width = 320, 320  # Increased from 64x64 to 320x320


# Define the function to get the inputs
def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


# Define the function to get the init inputs
def get_init_inputs():
    in_features = 25600  # Calculated as (320/2) * (320/2) = 160*160 = 25600
    out_features = 512
    max_pool_kernel_size = (2, 2)
    return [in_features, out_features, max_pool_kernel_size]