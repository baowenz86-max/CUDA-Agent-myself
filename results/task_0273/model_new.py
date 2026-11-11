import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, exponent):
        super().__init__()
        # Store pooling configuration (not used as nn module, just attributes)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # Exponent parameter (matches state_dict key 'pow')
        self.pow = nn.Parameter(torch.tensor(exponent))
        # Note: in_channels and out_channels are accepted but not used in this model
        # (reference model ignores them except in signature)

    def forward(self, x):
        x = cuda_extension.avgpool2d_forward(x, self.kernel_size, self.stride, self.padding)
        x = cuda_extension.pow_scalar_forward(x, self.pow)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.avgpool2d_forward(x, self.kernel_size, self.stride, self.padding)
        yield "step_001", x
        x = cuda_extension.pow_scalar_forward(x, self.pow)
        yield "step_002", x
        yield "step_003", x
