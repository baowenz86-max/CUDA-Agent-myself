import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, min_val, max_val):
        super().__init__()
        self.input_size = input_size
        self.min_val = float(min_val)
        self.max_val = float(max_val)

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        return cuda_extension.hardtanh_forward(x, min_val=self.min_val, max_val=self.max_val)

    def debug_forward(self, *inputs):
        x = inputs[0]
        if not x.is_cuda:
            x = x.cuda()
        output = cuda_extension.hardtanh_forward(x, min_val=self.min_val, max_val=self.max_val)
        yield 'step_001', output
