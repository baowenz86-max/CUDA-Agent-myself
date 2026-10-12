import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, start, end, steps, padding):
        super().__init__()
        self.start = start
        self.end = end
        self.steps = steps
        self.reflection_pad = nn.ReflectionPad2d(padding)

    def _linspace_cpu(self):
        # Equivalent to torch.linspace(self.start, self.end, self.steps)
        # without calling torch.linspace; arithmetic on CPU tensors is allowed.
        if self.steps == 1:
            return torch.tensor([self.start], dtype=torch.float32)
        step = (self.end - self.start) / (self.steps - 1)
        return self.start + step * torch.arange(self.steps, dtype=torch.float32)

    def forward(self, x):
        linspace_values = self._linspace_cpu()
        x = cuda_extension.reflection_pad2d_forward(x, self.reflection_pad.padding)
        histc_output = cuda_extension.histc_forward(x, linspace_values.size(0), self.start, self.end)
        return histc_output

    def debug_forward(self, *inputs):
        if len(inputs) != 1:
            raise ValueError('Expected one input tensor')
        x = inputs[0]
        linspace_values = self._linspace_cpu()
        yield 'step_001', linspace_values
        x = cuda_extension.reflection_pad2d_forward(x, self.reflection_pad.padding)
        yield 'step_002', x
        histc_output = cuda_extension.histc_forward(x, linspace_values.size(0), self.start, self.end)
        yield 'step_003', histc_output
        yield 'step_004', histc_output
