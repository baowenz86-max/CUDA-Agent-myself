import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

    def forward(self, x):
        x = torch.ops.cuda_extension.log_softmax_cuda(x, self.input_size)
        x = torch.ops.cuda_extension.tanh_cuda(x)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = torch.ops.cuda_extension.log_softmax_cuda(x, self.input_size)
        yield ('step_001', x)
        x = torch.ops.cuda_extension.tanh_cuda(x)
        yield ('step_002', x)
        yield ('step_003', x)
        return
