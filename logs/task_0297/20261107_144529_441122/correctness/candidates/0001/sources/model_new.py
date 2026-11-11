import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = cuda_extension.log_softmax_forward(x, 1)
        x = cuda_extension.triu_forward(x, 0)
        return x

    def debug_forward(self, x):
        x = cuda_extension.log_softmax_forward(x, 1)
        yield 'step_001', x
        x = cuda_extension.triu_forward(x, 0)
        yield 'step_002', x
        yield 'step_003', x
