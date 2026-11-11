import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, output_size):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, x):
        x = cuda_extension.custom_softmax(x, self.input_size)
        x = cuda_extension.custom_sub_scalar(x, self.output_size)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x_softmax = cuda_extension.custom_softmax(x, self.input_size)
        yield 'step_001', x_softmax
        x_sub = cuda_extension.custom_sub_scalar(x_softmax, self.output_size)
        yield 'step_002', x_sub
        yield 'step_003', x_sub
