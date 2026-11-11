import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max_pool2d(x)


batch_size = 16
in_channels = 3
height, width = 2560, 2560  # Increased from 512 to 2560 to increase computation


def get_inputs():
    # Create a random tensor of shape [batch_size, in_channels, height, width]
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]


def get_init_inputs():
    # No additional parameters needed for MaxPool2d
    return []