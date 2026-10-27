import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, n_class, chunks, dim_chunk):
        super().__init__()
        self.n_class = n_class
        self.chunks = chunks
        self.dim_chunk = dim_chunk

    def forward(self, x):
        x = cuda_extension.softmax_forward(x, dim=1)
        x, _ = cuda_extension.sort_forward(x, dim=1)
        x = cuda_extension.split_forward(x, self.dim_chunk, dim=1)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        softmax_out = cuda_extension.softmax_forward(x, dim=1)
        yield ("step_001", softmax_out)
        sorted_values, sorted_indices = cuda_extension.sort_forward(softmax_out, dim=1)
        yield ("step_002", (sorted_values, sorted_indices))
        split_tuple = cuda_extension.split_forward(sorted_values, self.dim_chunk, dim=1)
        yield ("step_003", split_tuple)
        yield ("step_004", split_tuple)
