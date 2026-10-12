import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, start, end, steps, padding):
        super().__init__()
        self.start = start
        self.end = end
        self.steps = steps
        # Preserve state_dict key 'reflection_pad' by constructing the same module.
        # ReflectionPad2d has no parameters, so it doesn't affect state_dict values.
        self.reflection_pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        # Step 1: linspace (non-CUDA path, allowed)
        linspace_values = torch.linspace(self.start, self.end, self.steps)
        # Step 2: reflection pad using custom CUDA op
        x = cuda_extension.reflection_pad2d_forward(x, self.reflection_pad.padding)
        # Step 3: histogram using custom CUDA op
        histc_output = cuda_extension.histc_forward(x, linspace_values.size(0), self.start, self.end)
        return histc_output

    def debug_forward(self, *inputs):
        if len(inputs) != 1:
            raise ValueError('Expected one input tensor')
        x = inputs[0]
        # Step 1
        linspace_values = torch.linspace(self.start, self.end, self.steps)
        yield 'step_001', linspace_values
        # Step 2
        x = cuda_extension.reflection_pad2d_forward(x, self.reflection_pad.padding)
        yield 'step_002', x
        # Step 3
        histc_output = cuda_extension.histc_forward(x, linspace_values.size(0), self.start, self.end)
        yield 'step_003', histc_output
        # Step 4 (final return)
        yield 'step_004', histc_output
