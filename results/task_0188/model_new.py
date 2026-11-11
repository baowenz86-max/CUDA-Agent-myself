import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        return cuda_extension.cummin_forward(x)

    def debug_forward(self, x):
        out = cuda_extension.cummin_forward(x)
        yield 'step_001', out
