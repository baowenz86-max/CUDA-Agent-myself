import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, norm_size, alpha, beta, k):
        super().__init__()
        self.norm_size = norm_size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        x = cuda_extension.lrn_forward(x, self.norm_size, self.alpha, self.beta, self.k)
        return x

    def debug_forward(self, x):
        out = cuda_extension.lrn_forward(x, self.norm_size, self.alpha, self.beta, self.k)
        yield ('step_001', out)
        yield ('step_002', out)
