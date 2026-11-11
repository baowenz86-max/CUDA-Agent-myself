import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, padding, value):
        super().__init__()
        # Reference ConstantPad1d accepts tuple, but custom pad_forward uses scalar symmetric padding.
        # Task always provides (10,10); extract the integer to match binding signature.
        if isinstance(padding, (tuple, list)):
            if padding[0] != padding[1]:
                raise ValueError('pad_forward only supports symmetric padding')
            self.padding = padding[0]
        else:
            self.padding = padding
        self.value = value

    def forward(self, x):
        x = cuda_extension.softmax_forward(x, dim=1)
        x = cuda_extension.pad_forward(x, padding=self.padding, value=self.value)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        step = 0
        yield ('step_001', cuda_extension.softmax_forward(x, dim=1))
        x = cuda_extension.softmax_forward(x, dim=1)
        step += 1
        yield ('step_002', cuda_extension.pad_forward(x, padding=self.padding, value=self.value))
        x = cuda_extension.pad_forward(x, padding=self.padding, value=self.value)
        step += 1
        yield ('step_003', x)
