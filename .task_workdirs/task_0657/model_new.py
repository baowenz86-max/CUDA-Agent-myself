import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x):
        x = cuda_extension.mean_dim1(x)
        x = cuda_extension.mean_dim2(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.mean_dim1(x)
        yield ('step_001', x)
        x = cuda_extension.mean_dim2(x)
        yield ('step_002', x)
        yield ('step_003', x)
        return x
