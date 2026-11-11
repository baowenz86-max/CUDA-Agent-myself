import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # No trainable parameters, same as reference

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.sigmoid_forward(x)
        x = cuda_extension.cosine_similarity_forward(x, y)
        return x

    def debug_forward(self, x: torch.Tensor, y: torch.Tensor):
        x_sigmoid = cuda_extension.sigmoid_forward(x)
        yield 'step_001', x_sigmoid
        x_cos = cuda_extension.cosine_similarity_forward(x_sigmoid, y)
        yield 'step_002', x_cos
        yield 'step_003', x_cos


batch_size = 512


def get_inputs():
    x = torch.randn(batch_size, 3, 256, 256)
    y = torch.randn(batch_size, 3, 256, 256)
    return [x, y]


def get_init_inputs():
    return []  # No parameters to initialize