import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_classes, depth, padding_size, padding_value):
        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.padding_size = padding_size
        self.padding_value = padding_value

    def forward(self, x):
        # Convert to one-hot encoding
        x = F.one_hot(x.long(), num_classes=self.num_classes)
        x = x.float()
        # Pad the tensor
        x = F.pad(
            x, (self.padding_size, self.padding_size, self.padding_size, self.padding_size), value=self.padding_value
        )
        # Apply softmax
        x = F.softmax(x, dim=self.depth)
        return x


batch_size = 512  # Increased to ensure at least one dimension >=512
classes = 5
height, width = 128, 128  # Increased spatial dimensions to increase computation
depth = 1
padding_size = 2
padding_value = 0.0


def get_inputs():
    return [torch.randint(0, classes, (batch_size, height, width))]


def get_init_inputs():
    return [classes, depth, padding_size, padding_value]