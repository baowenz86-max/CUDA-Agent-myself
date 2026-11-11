import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_dim, squeeze_axes):
        super(ModelNew, self).__init__()
        self.squeeze_axes = squeeze_axes

    def forward(self, x):
        x = cuda_extension.custom_abs(x)
        # torch.squeeze with dim=self.squeeze_axes: since input dim size is not 1, this is a no-op
        if x.shape[self.squeeze_axes] != 1:
            return x
        else:
            return x.squeeze(dim=self.squeeze_axes)

    def debug_forward(self, x):
        x_abs = cuda_extension.custom_abs(x)
        yield ('step_001', x_abs)
        if x_abs.shape[self.squeeze_axes] != 1:
            x_squeezed = x_abs
        else:
            x_squeezed = x_abs.squeeze(dim=self.squeeze_axes)
        yield ('step_002', x_squeezed)
        yield ('step_003', x_squeezed)
