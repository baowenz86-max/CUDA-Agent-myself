import torch
import torch.nn as nn
import cuda_extension


def _pack_int(value):
    if isinstance(value, (tuple, list)):
        assert len(value) == 2
        return (int(value[0]) << 32) | int(value[1])
    else:
        return (int(value) << 32) | 0xffffffff


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = cuda_extension.unfold_forward(
            x,
            _pack_int(self.unfold.kernel_size),
            _pack_int(self.unfold.dilation),
            _pack_int(self.unfold.padding),
            _pack_int(self.unfold.stride)
        )
        x = cuda_extension.cat_forward(x, x, dim=1)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.unfold_forward(
            x,
            _pack_int(self.unfold.kernel_size),
            _pack_int(self.unfold.dilation),
            _pack_int(self.unfold.padding),
            _pack_int(self.unfold.stride)
        )
        yield 'step_001', x
        x = cuda_extension.cat_forward(x, x, dim=1)
        yield 'step_002', x
        yield 'step_003', x
