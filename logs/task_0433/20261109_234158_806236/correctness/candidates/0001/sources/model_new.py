import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, kernel_size, dilation, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        kH, kW = self.kernel_size
        dH, dW = self.dilation
        sH, sW = self.stride if isinstance(self.stride, (tuple, list)) else (self.stride, self.stride)
        x = cuda_extension.custom_unfold_forward(
            x, kH, kW, dH, dW, sH, sW
        )
        N = x.size(0)
        C_out = x.size(1)
        L = x.size(2)
        x = cuda_extension.custom_reshape_3d(x, N, L, C_out)
        return x

    def debug_forward(self, x):
        kH, kW = self.kernel_size
        dH, dW = self.dilation
        sH, sW = self.stride if isinstance(self.stride, (tuple, list)) else (self.stride, self.stride)
        x = cuda_extension.custom_unfold_forward(
            x, kH, kW, dH, dW, sH, sW
        )
        yield "step_001", x
        N = x.size(0)
        C_out = x.size(1)
        L = x.size(2)
        x = cuda_extension.custom_reshape_3d(x, N, L, C_out)
        yield "step_002", x
        yield "step_003", x
