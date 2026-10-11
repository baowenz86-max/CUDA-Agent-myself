import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, kernel_size, padding_size):
        super().__init__()
        self.padding_size = padding_size

    def forward(self, x):
        return cuda_extension.reflect_pad_geq_remainder_forward(
            x,
            self.padding_size[0],
            self.padding_size[1],
            self.padding_size[2],
            self.padding_size[3],
        )
