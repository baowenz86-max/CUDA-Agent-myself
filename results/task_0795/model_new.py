import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size, num_elements):
        super().__init__()
        self.input_size = input_size
        self.num_elements = num_elements

    def forward(self, x):
        x = cuda_extension.custom_stack_relu(x, self.input_size, self.num_elements)
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_stack_relu(x, self.input_size, self.num_elements)
        yield 'step_001', x
        yield 'step_002', x
        yield 'step_003', x
