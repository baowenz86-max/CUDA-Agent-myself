import torch
import cuda_extension

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        abs_out = cuda_extension.abs_forward(x)
        out = cuda_extension.hardsigmoid_forward(abs_out)
        return out

    def debug_forward(self, x):
        abs_out = cuda_extension.abs_forward(x)
        yield 'step_001', abs_out
        out = cuda_extension.hardsigmoid_forward(abs_out)
        yield 'step_002', out
        yield 'step_003', out
