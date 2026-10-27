import torch
import torch.nn as nn

import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.cuda()
        x = cuda_extension.elu_forward(x, alpha=1.0)
        x = cuda_extension.digamma_forward(x)
        x = cuda_extension.triu_forward(x)
        return x

    def debug_forward(self, x):
        x = x.cuda()
        x = cuda_extension.elu_forward(x, alpha=1.0)
        yield 'step_001', x
        x = cuda_extension.digamma_forward(x)
        yield 'step_002', x
        x = cuda_extension.triu_forward(x)
        yield 'step_003', x
        yield 'step_004', x
