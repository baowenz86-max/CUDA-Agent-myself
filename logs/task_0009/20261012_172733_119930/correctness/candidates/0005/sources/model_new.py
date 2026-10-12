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

    def _make_linspace_cpu(self):
        # Manually construct linspace values without torch.linspace
        # Reference: torch.linspace(start, end, steps) on CPU, float32
        # We build a CPU tensor of size steps and fill each element.
        values = torch.empty(self.steps, dtype=torch.float32, device='cpu')
        if self.steps == 1:
            values[0] = self.start
        else:
            step_size = (self.end - self.start) / (self.steps - 1)
            for i in range(self.steps):
                values[i] = self.start + i * step_size
        return values

    def forward(self, x):
        linspace_values = self._make_linspace_cpu()
        x_padded = cuda_extension.reflection_pad2d_forward(x, self.padding)
        histc_output = cuda_extension.histc_forward(x_padded, bins=linspace_values.size(0), min=self.start, max=self.end)
        return histc_output

    def debug_forward(self, x):
        linspace_values = self._make_linspace_cpu()
        yield ('step_001', linspace_values)
        x_padded = cuda_extension.reflection_pad2d_forward(x, self.padding)
        yield ('step_002', x_padded)
        histc_output = cuda_extension.histc_forward(x_padded, bins=linspace_values.size(0), min=self.start, max=self.end)
        yield ('step_003', histc_output)
        yield ('step_004', histc_output)
