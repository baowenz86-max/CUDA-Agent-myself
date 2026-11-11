import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, size: int, alpha: float, beta: float, k: float):
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.cross_map_lrn_forward(x, self.size, self.alpha, self.beta, self.k)

    def debug_forward(self, *inputs):
        # Unpack inputs as per forward signature; in this case only one tensor
        x = inputs[0]
        y = cuda_extension.cross_map_lrn_forward(x, self.size, self.alpha, self.beta, self.k)
        yield 'step_001', y
