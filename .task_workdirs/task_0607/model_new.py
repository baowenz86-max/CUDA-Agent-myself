import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, channels, kernel_size, dilation, input_shape):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation)
        self.channels = channels
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x):
        unfold_abs = cuda_extension.custom_unfold_abs(
            x,
            self.kernel_size[0], self.kernel_size[1],
            self.dilation,
            0, 0,  # padding_h, padding_w
            1, 1   # stride_h, stride_w
        )
        return unfold_abs

    def debug_forward(self, *inputs):
        x = inputs[0]
        unfold_out = cuda_extension.custom_unfold_abs(
            x,
            self.kernel_size[0], self.kernel_size[1],
            self.dilation,
            0, 0,  # padding_h, padding_w
            1, 1   # stride_h, stride_w
        )
        yield 'step_001', unfold_out
        abs_out = cuda_extension.custom_unfold_abs(
            x,
            self.kernel_size[0], self.kernel_size[1],
            self.dilation,
            0, 0,  # padding_h, padding_w
            1, 1   # stride_h, stride_w
        )
        yield 'step_002', abs_out
        yield 'step_003', abs_out
