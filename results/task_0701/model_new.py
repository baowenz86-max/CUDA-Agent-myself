import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, split_size, num_splits, axis):
        super().__init__()
        self.split_size = split_size
        self.num_splits = num_splits
        self.axis = axis

    def forward(self, x):
        # Split the tensor along the specified axis using custom CUDA op (returns tuple of views)
        splits = cuda_extension.split(x, self.split_size, self.axis)

        # Sum across the splits using custom CUDA sum for each split
        sums = [cuda_extension.sum(t) for t in splits]

        return sums

    def debug_forward(self, *inputs):
        x = inputs[0]
        splits = cuda_extension.split(x, self.split_size, self.axis)
        yield "step_001", splits
        sums = [cuda_extension.sum(t) for t in splits]
        yield "step_002", sums
        yield "step_003", sums
