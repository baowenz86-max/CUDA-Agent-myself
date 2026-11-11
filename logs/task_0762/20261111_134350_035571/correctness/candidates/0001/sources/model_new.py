import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, start_range, end_range, pad_dims, pad_width, pad_mode, pad_value):
        super(ModelNew, self).__init__()
        self.pad_width = pad_width
        self.pad_mode = pad_mode
        self.pad_value = pad_value

    def forward(self, x):
        x = cuda_extension.custom_pad3d(
            x,
            self.pad_width[0],
            self.pad_width[1],
            self.pad_width[2],
            self.pad_width[3],
            self.pad_value,
        )
        return x

    def debug_forward(self, x):
        x = cuda_extension.custom_pad3d(
            x,
            self.pad_width[0],
            self.pad_width[1],
            self.pad_width[2],
            self.pad_width[3],
            self.pad_value,
        )
        yield "step_001", x
        yield "step_002", x
