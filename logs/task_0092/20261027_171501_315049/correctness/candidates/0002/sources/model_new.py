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
        x = cuda_extension.custom_softmax_forward(x, dim=1)
        sorted_result = cuda_extension.custom_sort_forward(x, dim=1)
        x = sorted_result.values  # 只取 values，与参考代码 x, _ = torch.sort(x, dim=1) 一致
        x = cuda_extension.custom_split_forward(x, self.dim_chunk, dim=1)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_softmax_forward(x, dim=1)
        yield 'step_001', x
        sorted_result = cuda_extension.custom_sort_forward(x, dim=1)
        x = sorted_result.values
        yield 'step_002', sorted_result
        x = cuda_extension.custom_split_forward(x, self.dim_chunk, dim=1)
        yield 'step_003', x
        yield 'step_004', x