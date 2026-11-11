import torch


class ModelNew(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return cuda_extension.var_mean_dim(x, self.dim, self.keepdim)

    def debug_forward(self, *inputs):
        result = self.forward(*inputs)
        yield 'step_001', result
