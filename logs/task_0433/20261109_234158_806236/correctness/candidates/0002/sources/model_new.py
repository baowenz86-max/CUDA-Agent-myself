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
        # Reference: torch.reshape(x, (x.size(0), -1, kernel_size[0]*kernel_size[1]))
        # Here kernel_size[0]*kernel_size[1] = C_out, and -1 resolves to N*L/C_out = N*L/C_out
        # Since C_out = 45, N*L = 64*128016 = 8193024, divided by 15 gives 384048.
        # The last dimension must be 15 (kernel area) not C_out.
        x = cuda_extension.custom_reshape_3d(x, N, N * L, C_out)
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
        x = cuda_extension.custom_reshape_3d(x, N, N * L, C_out)
        yield "step_002", x
        yield "step_003", x
