import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, device):
        super(ModelNew, self).__init__()
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = cuda_extension.custom_abs(x)
        return x

    def debug_forward(self, x):
        x = x.to(self.device)
        yield 'step_001', x
        x = cuda_extension.custom_abs(x)
        yield 'step_002', x
        yield 'step_003', x
