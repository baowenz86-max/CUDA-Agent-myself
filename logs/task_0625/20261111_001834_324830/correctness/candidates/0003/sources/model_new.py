import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.sum_channel_dim(x)
        x = cuda_extension.clip_01(x)
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.sum_channel_dim(x)
        yield ("step_001", x)
        x = cuda_extension.clip_01(x)
        yield ("step_002", x)
        yield ("step_003", x)
