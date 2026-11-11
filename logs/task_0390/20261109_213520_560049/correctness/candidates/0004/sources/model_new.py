import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, groups, block_size):
        super().__init__()
        self.num_channels = num_channels
        self.groups = groups
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.channel_shuffle_forward(x, self.groups)

    def debug_forward(self, *inputs):
        x = inputs[0]
        output = cuda_extension.channel_shuffle_forward(x, self.groups)
        yield 'step_001', output
