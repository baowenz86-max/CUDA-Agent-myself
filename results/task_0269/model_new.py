import torch
import torch.nn as nn
import cuda_extension

class ModelNew(nn.Module):
    def __init__(self, norm_type: int, kernel_size: tuple, stride: int = 1, ceil_mode: bool = False):
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.lp_pool2d_forward(
            x, self.norm_type, self.kernel_size[0], self.kernel_size[1],
            self.stride, 0, 0, self.ceil_mode
        )

    def debug_forward(self, x: torch.Tensor):
        y = self.forward(x)
        yield 'step_001', y
