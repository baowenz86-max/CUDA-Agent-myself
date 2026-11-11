import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, channels, padding_size, min_value):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.min_value = min_value

    def forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(
            x,
            self.reflection_pad.padding[0],
            self.reflection_pad.padding[1],
            self.reflection_pad.padding[2],
            self.reflection_pad.padding[3],
        )
        return cuda_extension.clamp_min_forward(x, self.min_value)

    def debug_forward(self, x):
        x = cuda_extension.reflection_pad2d_forward(
            x,
            self.reflection_pad.padding[0],
            self.reflection_pad.padding[1],
            self.reflection_pad.padding[2],
            self.reflection_pad.padding[3],
        )
        yield 'step_001', x
        x = cuda_extension.clamp_min_forward(x, self.min_value)
        yield 'step_002', x
