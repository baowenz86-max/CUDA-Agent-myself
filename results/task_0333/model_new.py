import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, num_channels: int, size: int, alpha: float, beta: float, k: float):
        super().__init__()
        self.local_response_norm = nn.LocalResponseNorm(size, alpha, beta, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_extension.lrn_forward(x, self.local_response_norm.size, self.local_response_norm.alpha, self.local_response_norm.beta, self.local_response_norm.k)

    def debug_forward(self, *inputs):
        x = inputs[0]
        y = cuda_extension.lrn_forward(x, self.local_response_norm.size, self.local_response_norm.alpha, self.local_response_norm.beta, self.local_response_norm.k)
        yield 'step_001', y
