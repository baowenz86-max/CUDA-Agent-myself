import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.subtract_mean = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        mean = cuda_extension.channel_mean_forward(x)
        return cuda_extension.subtract_mean_forward(x, mean, self.subtract_mean.item())

    def debug_forward(self, x):
        mean = cuda_extension.channel_mean_forward(x)
        yield "step_001", mean
        out = cuda_extension.subtract_mean_forward(x, mean, self.subtract_mean.item())
        yield "step_002", out
