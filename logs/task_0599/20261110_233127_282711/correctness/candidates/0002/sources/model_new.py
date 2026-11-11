import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, param_list):
        super().__init__()
        self.param_list = nn.ParameterList(param_list)

    def forward(self, x):
        x = cuda_extension.abs_forward(x)
        for param in self.param_list:
            x = cuda_extension.add_scalar_tensor_forward(x, param)
        return x

    def debug_forward(self, x):
        x = cuda_extension.abs_forward(x)
        for param in self.param_list:
            x = cuda_extension.add_scalar_tensor_forward(x, param)
        yield 'final', x
