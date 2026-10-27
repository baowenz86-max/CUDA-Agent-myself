import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_class, chunks, dim_chunk):
        super().__init__()
        self.n_class = n_class
        self.chunks = chunks
        self.dim_chunk = dim_chunk

    def forward(self, x):
        x = torch.softmax(x, dim=1)
        x, _ = torch.sort(x, dim=1)
        x = torch.split(x, self.dim_chunk, dim=1)
        return x


n_class = 10
batch_size = 512  # Increased from 64 to 512 to ensure first dimension is >=512
height, width = 64, 64  # Increased from 32x32 to 64x64 to increase computation
chunks = 2
dim_chunk = 3


def get_inputs():
    return [torch.randn(batch_size, n_class, height, width)]


def get_init_inputs():
    return [n_class, chunks, dim_chunk]