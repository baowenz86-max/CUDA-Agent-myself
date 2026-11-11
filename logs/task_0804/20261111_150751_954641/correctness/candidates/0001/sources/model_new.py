import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1, x2):
        return cuda_extension.stack_forward(x1, x2)

    def debug_forward(self, x1, x2):
        out = cuda_extension.stack_forward(x1, x2)
        yield 'step_001', out
