import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, channels, padding_size):
        super().__init__()
        self.channels = channels
        self.padding_size = padding_size

    def forward(self, x):
        x = cuda_extension.pad_forward(
            x,
            self.padding_size[0],
            self.padding_size[1],
            self.padding_size[2],
            self.padding_size[3],
        )
        x = cuda_extension.mean_dim_forward(x, self.channels)
        return x

    def debug_forward(self, x):
        x = cuda_extension.pad_forward(
            x,
            self.padding_size[0],
            self.padding_size[1],
            self.padding_size[2],
            self.padding_size[3],
        )
        yield "step_001", x
        x = cuda_extension.mean_dim_forward(x, self.channels)
        yield "step_002", x
        yield "step_003", x
