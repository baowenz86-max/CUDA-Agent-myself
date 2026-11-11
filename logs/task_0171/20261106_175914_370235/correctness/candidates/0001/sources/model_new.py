import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, x):
        # Input is expected to be on CUDA; if not, move it (as reference would via device placement)
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.log_softmax_dim1(x)
        x = cuda_extension.elu(x)
        return x

    def debug_forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        x = cuda_extension.log_softmax_dim1(x)
        yield ('step_001', x)
        x = cuda_extension.elu(x)
        yield ('step_002', x)
        yield ('step_003', x)
