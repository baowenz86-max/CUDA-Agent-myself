import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, padding_size):
        super().__init__()
        # Keep the same module as reference for state_dict compatibility;
        # it is not used in forward, since forward must use the custom CUDA op.
        self.reflection_pad = nn.ReflectionPad2d(padding_size)
        self.padding_top = padding_size[0]
        self.padding_bottom = padding_size[1]
        self.padding_left = padding_size[2]
        self.padding_right = padding_size[3]

    def forward(self, x):
        # Call custom CUDA reflection padding operator
        x = cuda_extension.reflection_pad2d_forward(
            x,
            self.padding_top,
            self.padding_bottom,
            self.padding_left,
            self.padding_right
        )
        # torch.detach is a non-computational path, keep as native call
        x = torch.detach(x)
        return x

    def debug_forward(self, *inputs):
        # Yield checkpoints in order
        x = inputs[0]
        x_padded = cuda_extension.reflection_pad2d_forward(
            x,
            self.padding_top,
            self.padding_bottom,
            self.padding_left,
            self.padding_right
        )
        yield 'step_001', x_padded
        x_detached = torch.detach(x_padded)
        yield 'step_002', x_detached
        yield 'step_003', x_detached
