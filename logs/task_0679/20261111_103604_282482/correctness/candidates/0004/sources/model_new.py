import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        sum_result = cuda_extension.torch_sum_dim1(x)
        argmax_result = cuda_extension.torch_argmax_all(sum_result)
        return argmax_result

    def debug_forward(self, x):
        sum_result = cuda_extension.torch_sum_dim1(x)
        yield ('step_001', sum_result)
        argmax_result = cuda_extension.torch_argmax_all(sum_result)
        yield ('step_002', argmax_result)
        yield ('step_003', argmax_result)
