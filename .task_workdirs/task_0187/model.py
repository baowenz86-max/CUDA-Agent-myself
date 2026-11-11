import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.softmax(x, dim=1)
        x = F.hardtanh(x)
        return x


# Define a function to generate random inputs
def get_inputs():
    batch_size, num_classes, height, width = 512, 10, 224, 224  # Increased height/width to 224
    x = torch.randn(batch_size, num_classes, height, width)
    return [x]


# Define a function to get the initialization inputs
def get_init_inputs():
    return []  # No initialization required for this model