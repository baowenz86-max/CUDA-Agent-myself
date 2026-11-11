import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, channels: int, kernel_size: tuple, stride: int = 1, padding: tuple = (0, 0)):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = cuda_extension.channel_mean_forward(x)
        x = cuda_extension.max_pool2d_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride,
            self.stride,
            self.padding[0],
            self.padding[1],
        )
        return x

    def debug_forward(self, x: torch.Tensor):
        x = cuda_extension.channel_mean_forward(x)
        yield "step_001", x
        x = cuda_extension.max_pool2d_forward(
            x,
            self.kernel_size[0],
            self.kernel_size[1],
            self.stride,
            self.stride,
            self.padding[0],
            self.padding[1],
        )
        yield "step_002", x
        yield "step_003", x
