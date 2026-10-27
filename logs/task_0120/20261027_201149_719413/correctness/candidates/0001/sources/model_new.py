import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, kernel_size, dilation, padding):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding

    def forward(self, x):
        # Move input to CUDA if it is on CPU (reference execution device is CUDA)
        if x.device.type != 'cuda':
            x = x.cuda()
        x = cuda_extension.unfold_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.dilation[0],
            self.dilation[1],
            self.padding[0],
            self.padding[1]
        )
        x = cuda_extension.sign_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        if x.device.type != 'cuda':
            x = x.cuda()
        unfolded = cuda_extension.unfold_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.dilation[0],
            self.dilation[1],
            self.padding[0],
            self.padding[1]
        )
        yield ('step_001', unfolded)
        signed = cuda_extension.sign_forward(unfolded)
        yield ('step_002', signed)
        yield ('step_003', signed)
