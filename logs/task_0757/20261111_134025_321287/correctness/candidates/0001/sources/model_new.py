import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, num_inputs, num_classes):
        super(ModelNew, self).__init__()

    def forward(self, x1, x2):
        x = cuda_extension.custom_cat(x1, x2, 1)
        x = cuda_extension.custom_ne(x, 0.0)
        return x

    def debug_forward(self, x1, x2):
        x = cuda_extension.custom_cat(x1, x2, 1)
        yield 'step_001', x
        x = cuda_extension.custom_ne(x, 0.0)
        yield 'step_002', x
        yield 'step_003', x
