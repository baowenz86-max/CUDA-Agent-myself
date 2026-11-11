import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_inputs, bias_shape):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, tensor1, tensor2):
        x = torch.addcdiv(x, tensor1, tensor2)
        x = x + self.bias
        return x


batch_size = 512
num_inputs = 5
height, width = 256, 256  # Increased spatial dimensions
bias_shape = (num_inputs, 1, 1)


def get_inputs():
    tensor1 = torch.randn(batch_size, num_inputs, height, width)
    tensor2 = torch.randn(batch_size, num_inputs, height, width)
    x = torch.randn(batch_size, num_inputs, height, width)
    return [x, tensor1, tensor2]


def get_init_inputs():
    return [num_inputs, bias_shape]