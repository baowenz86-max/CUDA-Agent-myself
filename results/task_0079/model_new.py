import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        self.padding_size = padding_size

    def forward(self, x):
        # Move input to CUDA if not already (using tensor property, not torch module)
        if not x.is_cuda:
            x = x.cuda()
        # ReflectionPad2d with padding tuple (left, right, top, bottom)
        left, right, top, bottom = self.padding_size
        x = cuda_extension.pad_forward(x, left, right, top, bottom)
        # Element-wise square
        x = cuda_extension.square_forward(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        left, right, top, bottom = self.padding_size
        x = cuda_extension.pad_forward(x, left, right, top, bottom)
        yield 'step_001', x
        x = cuda_extension.square_forward(x)
        yield 'step_002', x
        yield 'step_003', x
        # The generator does not return anything explicitly; the last yield provides the final value.
