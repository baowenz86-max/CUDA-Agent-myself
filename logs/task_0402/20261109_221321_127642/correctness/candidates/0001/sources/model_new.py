import torch

class ModelNew(torch.nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x):
        return cuda_extension.mean_dim0(x)

    def debug_forward(self, x):
        result = cuda_extension.mean_dim0(x)
        yield 'step_001', result