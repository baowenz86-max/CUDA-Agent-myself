import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, n_class, chunks, dim_chunk):
        super().__init__()
        self.n_class = n_class
        self.chunks = chunks
        self.dim_chunk = dim_chunk

    def forward(self, x):
        x = self._move_to_cuda(x)
        x = custom_softmax_dim1(x)
        x = custom_sort_dim1(x, return_key=False)
        out = custom_split_dim1(x, self.dim_chunk)
        return out

    def debug_forward(self, x):
        x = self._move_to_cuda(x)
        # step_001: softmax
        x = custom_softmax_dim1(x)
        yield 'step_001', x
        # step_002: sort (need full tuple with indices)
        sort_result = custom_sort_dim1(x, return_key=True)
        yield 'step_002', sort_result
        x = sort_result[0]
        # step_003: split
        out = custom_split_dim1(x, self.dim_chunk)
        yield 'step_003', out
        # step_004: return x
        yield 'step_004', out

    def _move_to_cuda(self, x):
        if not x.is_cuda:
            x = x.to('cuda')
        return x


import cuda_extension

# These names will be resolved at runtime from cuda_extension
custom_softmax_dim1 = cuda_extension.custom_softmax_dim1
custom_sort_dim1 = cuda_extension.custom_sort_dim1
custom_split_dim1 = cuda_extension.custom_split_dim1
