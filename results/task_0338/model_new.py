import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        return cuda_extension.custom_ge_forward(x, self.threshold)

    def debug_forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        out = cuda_extension.custom_ge_forward(x, self.threshold)
        yield 'step_001', out
