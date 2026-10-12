import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, start, end, steps, padding):
        super().__init__()
        self.start = start
        self.end = end
        self.steps = steps
        self.padding = padding

    def forward(self, x):
        linspace_values = torch.linspace(self.start, self.end, self.steps)
        x_padded = cuda_extension.reflection_pad2d_forward(x, self.padding)
        histc_output = cuda_extension.histc_forward(x_padded, bins=linspace_values.size(0),
                                                    min=self.start, max=self.end)
        return histc_output

    def debug_forward(self, x):
        linspace_values = torch.linspace(self.start, self.end, self.steps)
        yield 'step_001', linspace_values
        x_padded = cuda_extension.reflection_pad2d_forward(x, self.padding)
        yield 'step_002', x_padded
        histc_output = cuda_extension.histc_forward(x_padded, bins=linspace_values.size(0),
                                                    min=self.start, max=self.end)
        yield 'step_003', histc_output
        yield 'step_004', histc_output
