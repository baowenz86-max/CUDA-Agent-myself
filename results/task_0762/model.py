import torch
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, start_range, end_range, pad_dims, pad_width, pad_mode, pad_value):
        super(Model, self).__init__()
        self.arange = torch.arange(start_range, end_range)
        self.pad_dims = pad_dims
        self.pad_width = pad_width
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def forward(self, x):
        x = F.pad(x, self.pad_width, self.pad_mode, self.pad_value)
        return x


batch_size = 200
channels = 1
height = 1024
width = 1024


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [0, 10, ((0, 1), (1, 2)), (2, 1, 1, 2), "constant", 0]