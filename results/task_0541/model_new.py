import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, num_features, eps, momentum):
        super().__init__()
        # Keep the exact same submodules as reference to preserve state_dict keys
        self.bn = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = cuda_extension.custom_batchnorm2d_eval(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps
        )
        x = cuda_extension.custom_gelu_forward(x)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_batchnorm2d_eval(
            x,
            self.bn.weight,
            self.bn.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps
        )
        yield ('step_001', x)
        x = cuda_extension.custom_gelu_forward(x)
        yield ('step_002', x)
        yield ('step_003', x)
