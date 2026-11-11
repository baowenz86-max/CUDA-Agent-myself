import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_size, index_size, dim, src_size):
        super().__init__()
        self.input_size = input_size
        self.index_size = index_size
        self.dim = dim
        self.src_size = src_size

    def forward(self, input_tensor, index, src):
        mean_value = torch.mean(input_tensor, dim=self.dim, keepdim=True)
        scattered_tensor = torch.scatter(input_tensor, self.dim, index, src)
        return mean_value, scattered_tensor


batch_size = 512
input_channels = 32
height, width = 128, 128  # Increased from 32x32 to 128x128 to increase computation
dim = 1
index_size = (batch_size, 1, height, width)
src_size = (batch_size, 1, height, width)


def get_inputs():
    input_tensor = torch.randn(batch_size, input_channels, height, width)
    index = torch.randint(0, input_channels, index_size)
    src = torch.randn(src_size)
    return [input_tensor, index, src]


def get_init_inputs():
    return [input_channels, index_size, dim, src_size]