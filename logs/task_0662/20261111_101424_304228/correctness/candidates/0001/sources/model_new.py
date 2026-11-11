import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, mean_dim, padding_amount):
        super().__init__()
        self.mean_dim = mean_dim
        self.padding_amount = padding_amount

    def forward(self, x):
        # Ensure input is CUDA and float32 (verifier provides CUDA tensor; this is for safety)
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        # Apply ZeroPad2d padding using custom op
        pad_left, pad_right, pad_top, pad_bottom = self.padding_amount
        x = cuda_extension.custom_pad2d(x, pad_left, pad_right, pad_top, pad_bottom)
        # Compute mean over specified dimension using custom op
        x = cuda_extension.custom_mean_dim(x, self.mean_dim)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
        pad_left, pad_right, pad_top, pad_bottom = self.padding_amount
        x = cuda_extension.custom_pad2d(x, pad_left, pad_right, pad_top, pad_bottom)
        yield 'step_001', x
        x = cuda_extension.custom_mean_dim(x, self.mean_dim)
        yield 'step_002', x
        yield 'step_003', x
