import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, padding, value):
        super().__init__()
        self.padding = padding
        self.value = value

    def forward(self, x):
        # Move to CUDA (execution device is cuda)
        x = x.cuda()
        # Step 1: argsort along last dim
        x = cuda_extension.argsort_forward(x)
        # Step 2: cast to float and constant pad
        x = cuda_extension.constant_pad1d_forward(
            x, pad_left=self.padding[0], pad_right=self.padding[1], value=self.value
        )
        # Step 3: log softmax along last dim
        x = cuda_extension.log_softmax_forward(x)
        return x

    def debug_forward(self, x):
        # Same computation but yield intermediate checkpoints
        x = x.cuda()
        x = cuda_extension.argsort_forward(x)
        yield 'step_001', x
        x = cuda_extension.constant_pad1d_forward(
            x, pad_left=self.padding[0], pad_right=self.padding[1], value=self.value
        )
        yield 'step_002', x
        x = cuda_extension.log_softmax_forward(x)
        yield 'step_003', x
        yield 'step_004', x
