import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, padding, value):
        super().__init__()
        self.padding = padding
        self.value = value
        self.pad1d = nn.ConstantPad1d(padding, value)

    def forward(self, x):
        x = torch.argsort(x, dim=-1)
        x = self.pad1d(x.float())  # Convert to float because ConstantPad1d requires floating point inputs
        x = F.log_softmax(x, dim=-1)
        return x


batch_size = 128
channels = 64
sequence_length = 2048  # Increased from 512 to increase computation time
padding = (16, 16)
value = 0.0


def get_inputs():
    return [torch.randn(batch_size, channels, sequence_length)]


def get_init_inputs():
    return [padding, value]