import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels, eps_val=1e-05, momentum_val=0.1):
        super().__init__()
        self.arange_tensor = torch.arange(num_channels)
        self.batchnorm = nn.BatchNorm2d(num_channels, eps=eps_val, momentum=momentum_val)

    def forward(self, x):
        x = cuda_extension.batchnorm2d_eval(
            x,
            self.batchnorm.weight,
            self.batchnorm.bias,
            self.batchnorm.running_mean,
            self.batchnorm.running_var,
            self.batchnorm.eps,
        )
        return x

    def debug_forward(self, x):
        x = self.forward(x)
        return [("step_001", x), ("step_002", x)]
