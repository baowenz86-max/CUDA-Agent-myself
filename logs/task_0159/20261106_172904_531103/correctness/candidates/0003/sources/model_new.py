import torch
import cuda_extension


class ModelNew(torch.nn.Module):
    def __init__(self, input_size, pad_width, pad_value):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.pad_width = pad_width
        self.pad_value = pad_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.view_as_real(x)
        x = cuda_extension.constant_pad(x, self.pad_width, self.pad_value)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.view_as_real(x)
        yield 'step_001', x
        x = cuda_extension.constant_pad(x, self.pad_width, self.pad_value)
        yield 'step_002', x
        yield 'step_003', x
