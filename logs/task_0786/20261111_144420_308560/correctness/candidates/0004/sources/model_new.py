import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        self._normalized_shape_tensor = torch.tensor(normalized_shape, dtype=torch.int64)

    def forward(self, x):
        x = cuda_extension.custom_mean(x)
        x = cuda_extension.custom_layer_norm(x, self._normalized_shape_tensor)
        return x

    def debug_forward(self, *inputs):
        x = inputs[0]
        x = cuda_extension.custom_mean(x)
        yield 'step_001', x
        x = cuda_extension.custom_layer_norm(x, self._normalized_shape_tensor)
        yield 'step_002', x
        yield 'step_003', x
