import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scale_factor):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.scale_factor = scale_factor
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bias = nn.Parameter(torch.randn(hidden_size))

    def forward(self, x):
        x = torch.addmm(self.bias, x, self.weight.t())
        x = torch.unsqueeze(x, 0)  # Adding a batch dimension
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="linear", align_corners=False)
        return x


batch_size = 8192
input_size = 128
hidden_size = 1024
output_size = 512
scale_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, output_size, scale_factor]