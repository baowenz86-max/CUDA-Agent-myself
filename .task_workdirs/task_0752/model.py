import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs

    def forward(self, *x):
        x = torch.cat(x, dim=1)
        x = torch.erf(x)
        return x


# Define hyperparameters for input shape
batch_size = 512
channels = 8
height = 160
width = 160


# Define the function to generate the input data
def get_inputs():
    tensor1 = torch.randn(batch_size, channels, height, width)
    tensor2 = torch.randn(batch_size, channels, height, width)
    return [tensor1, tensor2]


# Define the function to get the initialization parameters
def get_init_inputs():
    return [2, 16]