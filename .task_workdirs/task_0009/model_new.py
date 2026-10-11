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
        return cuda_extension.reflection_pad_histc_forward(
            x,
            self.padding,
            float(self.start),
            float(self.end),
            int(self.steps)
        )
