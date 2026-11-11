import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, sync_bn):
        super().__init__()
        self.sync_bn = sync_bn
        # Create parameters to match state_dict keys (norm.weight, norm.bias, etc.)
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(num_channels))
        self.norm.bias = nn.Parameter(torch.zeros(num_channels))
        self.norm.running_mean = torch.zeros(num_channels)
        self.norm.running_var = torch.ones(num_channels)
        self.norm.num_batches_tracked = torch.tensor(0, dtype=torch.long)
        self.norm.eps = 1e-5

    def forward(self, x):
        x = cuda_extension.view_as_real_forward(x)
        x = cuda_extension.sync_batchnorm_forward(x, self.norm.weight, self.norm.bias, self.norm.eps)
        return x

    def debug_forward(self, *inputs):
        x = cuda_extension.view_as_real_forward(inputs[0])
        yield 'step_001', x
        x = cuda_extension.sync_batchnorm_forward(x, self.norm.weight, self.norm.bias, self.norm.eps)
        yield 'step_002', x
        yield 'step_003', x
