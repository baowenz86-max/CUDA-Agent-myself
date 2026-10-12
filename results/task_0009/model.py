import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, start, end, steps, padding):
        super().__init__()
        self.start = start
        self.end = end
        self.steps = steps
        self.reflection_pad = nn.ReflectionPad2d(padding)

    def forward(self, x):
        linspace_values = torch.linspace(self.start, self.end, self.steps)
        x = self.reflection_pad(x)
        histc_output = torch.histc(x, bins=linspace_values.size(0), min=self.start, max=self.end)
        return histc_output


batch_size = 128
channels = 3
height = 512
width = 128  # Increased from 64 to 128 to increase computational load
start = 0.0
end = 10.0
steps = 20
padding = 2


def get_inputs():
    return [torch.randn(batch_size, channels, height, width)]


def get_init_inputs():
    return [start, end, steps, padding]