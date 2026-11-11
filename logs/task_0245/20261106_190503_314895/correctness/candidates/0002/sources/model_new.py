import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, padding_size, mean, std):
        super().__init__()
        self.padding_size = padding_size
        self.normal_layer = torch.nn.Parameter(
            torch.tensor(torch.normal(mean, std, size=(1, 1, 1, 1))), requires_grad=False
        )

    def forward(self, x):
        import cuda_extension
        # padding_size is a tuple (pad_left, pad_right, pad_top, pad_bottom), but symmetric so use single int
        pad = self.padding_size[0]
        x = cuda_extension.reflection_pad2d_forward(x, pad)
        x = cuda_extension.add_scalar_forward(x, self.normal_layer)
        return x

    def debug_forward(self, x):
        import cuda_extension
        pad = self.padding_size[0]
        x = cuda_extension.reflection_pad2d_forward(x, pad)
        yield "step_001", x
        x = cuda_extension.add_scalar_forward(x, self.normal_layer)
        yield "step_002", x
        yield "step_003", x
