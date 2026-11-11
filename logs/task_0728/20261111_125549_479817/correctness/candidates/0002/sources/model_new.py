import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self, input_size_1, input_size_2, squeeze_dim):
        super().__init__()
        self.squeeze_dim = squeeze_dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = cuda_extension.squeeze_dim(x1, self.squeeze_dim)
        x2 = cuda_extension.squeeze_dim(x2, self.squeeze_dim)
        return cuda_extension.concat_dim((x1, x2), self.squeeze_dim)

    def debug_forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = cuda_extension.squeeze_dim(x1, self.squeeze_dim)
        yield 'step_001', x1
        x2 = cuda_extension.squeeze_dim(x2, self.squeeze_dim)
        yield 'step_002', x2
        out = cuda_extension.concat_dim((x1, x2), self.squeeze_dim)
        yield 'step_003', out
